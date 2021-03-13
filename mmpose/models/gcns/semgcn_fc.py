import torch.nn as nn
import torch
from .semgcn_utils import *
from .semgcn import _ResGraphConv_Attention,SemGraphConv,_GraphConv
from scipy import sparse as sp
import numpy as np

# import registor
from ..registry import GCNS


@GCNS.register_module()
class SemGCN_FC(nn.Module):
    
    def __init__(self, adj=None, num_joints=17, hid_dim=None, coords_dim=(2, 2), p_dropout=None):
        '''
        这里的adj就是人体骨架的连接顺序
        :param adj:  adjacency matrix using for
        :param hid_dim:
        :param coords_dim:
        :param num_layers:
        :param nodes_group:
        :param p_dropout:
        '''
        super().__init__()
        # 这个heatmap_generator是用backbone网络的不同阶段feature生成最后的heatmap, 和不同尺度的feature
        self.heat_map_generator =HM_Extrect(num_joints)

        self.adj = self._build_adj_mx_from_edges(num_joints=num_joints, edge=adj)
        # self.adj_matrix 就是 self.adj
        adj = self.adj_matrix
        self.num_joints = num_joints

        self.gconv_input = _GraphConv(adj, coords_dim[0], hid_dim[0], p_dropout=p_dropout)
        # in here we set 4 gcn model in this part
        self.gconv_layers1 = _ResGraphConv_Attention(adj, hid_dim[0], hid_dim[1], hid_dim[0], p_dropout=p_dropout)
        self.gconv_layers2 = _ResGraphConv_Attention(adj, hid_dim[1]+256, hid_dim[2]+256, hid_dim[1]+256, p_dropout=p_dropout)
        self.gconv_layers3 = _ResGraphConv_Attention(adj, hid_dim[2]+512, hid_dim[3]+512, hid_dim[2]+256, p_dropout=p_dropout)
        self.gconv_layers4 = _ResGraphConv_Attention(adj, hid_dim[3]+640, hid_dim[4]+640, hid_dim[3]+256, p_dropout=p_dropout)

        self.gconv_output1 = SemGraphConv(hid_dim[1]+256, coords_dim[1], adj)
        self.gconv_output2 = SemGraphConv(hid_dim[2]+512, coords_dim[1], adj)
        self.gconv_output3 = SemGraphConv(hid_dim[3]+640, coords_dim[1], adj)

        #FC
        self.FC = nn.Sequential(nn.Sequential(make_fc(3072,1024),nn.ReLU(inplace=True)),nn.Sequential(make_fc(1024,1024),nn.ReLU(inplace=True)),make_fc(1024,2))

    @property
    def adj_matrix(self):
        return self.adj

    @adj_matrix.setter
    def adj_matrix(self,adj_matrix):
        self.adj = adj_matrix

    def _build_adj_mx_from_edges(self,num_joints,edge):
        def adj_mx_from_edges(num_pts, edges, sparse=True):

            edges = np.array(edges, dtype=np.int32)
            data, i, j = np.ones(edges.shape[0]), edges[:, 0], edges[:, 1]
            adj_mx = sp.coo_matrix((data, (i, j)), shape=(num_pts, num_pts), dtype=np.float32)

            # build symmetric adjacency matrix
            adj_mx = adj_mx + adj_mx.T.multiply(adj_mx.T > adj_mx) - adj_mx.multiply(adj_mx.T > adj_mx)
            adj_mx = normalize(adj_mx + sp.eye(adj_mx.shape[0]))
            if sparse:
                adj_mx = sparse_mx_to_torch_sparse_tensor(adj_mx)
            else:
                adj_mx = torch.tensor(adj_mx.todense(), dtype=torch.float)
            return adj_mx

        def sparse_mx_to_torch_sparse_tensor(sparse_mx):
            """Convert a scipy sparse matrix to a torch sparse tensor."""
            sparse_mx = sparse_mx.tocoo().astype(np.float32)
            indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
            values = torch.from_numpy(sparse_mx.data)
            shape = torch.Size(sparse_mx.shape)
            return torch.sparse.FloatTensor(indices, values, shape)

        def normalize(mx):
            """Row-normalize sparse matrix"""
            rowsum = np.array(mx.sum(1))
            r_inv = np.power(rowsum, -1).flatten()
            r_inv[np.isinf(r_inv)] = 0.
            r_mat_inv = sp.diags(r_inv)
            mx = r_mat_inv.dot(mx)
            return mx

        return adj_mx_from_edges(num_joints, edge, False)

    def forward(self, x, ret_features, hm_4=None):
        '''
        x: backbone网络检测的关节点坐标
        hm_4: 输入的用提前检测到的关节点坐标, 如果给了的话就直接用, 不再需要FC来回归出来
        ret_feature: 上采样的不同尺度的featuremap
        '''
        x = torch.from_numpy(x).cuda()
        # heatmap_generator是对backbone的feature进行反卷机上采样
        # results是不同尺度的feature经过反卷积的结果, heatmap是最后一层的输出
        results, heat_map = self.heat_map_generator(ret_features)
        # print('heatmap:', heat_map.shape) [16, 17, 64, 48]
        bs = heat_map.shape[0]
        # 这是通过fc层回归出关节点的坐标
        # TODO 积分英文 `integral`
        heat_map_intergral = self.FC(heat_map.view(bs*self.num_joints, -1)).view(bs, self.num_joints*2)
        # print('heatmap_integral: ', heat_map_intergral.shape) # [N, 34]

        # hm_4是从heatmap上积分得到的坐标值, 然后从原始feature上面sample到feature
        hm_4 = heat_map_intergral.view(-1, self.num_joints, 2)
        # print('result[0]:', results[0].shape) [16, 256, 16, 12]
        j_1_16 = F.grid_sample(results[0],hm_4[:,None,:,:], align_corners=False).squeeze(2)
        # print('j_1_16:', j_1_16.shape) [16, 256, 17]
        j_1_8 = F.grid_sample(results[1],hm_4[:,None,:,:], align_corners=False).squeeze(2)
        # print('j_1_8:', j_1_8.shape) [16, 256, 17]
        j_1_4 = F.grid_sample(results[2],hm_4[:,None,:,:], align_corners=False).squeeze(2)
        # print('j_1_4:', j_1_4.shape) [16, 128, 17]

        # x = torch.cat([hm_4,score],-1)
        # 用之前的SPPE得到的检测Pose作为初始pose, 然后再与回归得到的坐标feature进行连接, 一起回归出来最终的坐标
        out = self.gconv_input(hm_4)
        # out = self.gconv_input(x)
        # print('out0: ', out.shape) [16, 17, 128]
        out = self.gconv_layers1(out, None)
        # print('out1: ', out.shape) [16, 17, 128]
        out = self.gconv_layers2(out, j_1_16) # [16, 17, 128] [16, 256, 17]
        # print('out1: ', out.shape) # [16, 17, 384]
        out1 = self.gconv_output1(out) # [16, 17, 2]

        out = self.gconv_layers3(out, j_1_8) # [16, 17, 384] [16, 256, 17]
        # print('out2: ', out.shape) # [16, 17, 640]
        out2 = self.gconv_output2(out)

        out = self.gconv_layers4(out, j_1_4)
        # [16, 17, 768]
        out3 = self.gconv_output3(out)

        # print('out1: ', out1.size(), 'out2: ', out2.size(), 'out3: ', out3.size(), 'integral: ', heat_map_intergral.size())
        # out1:  torch.Size([16, 17, 2]) out2:  torch.Size([16, 17, 2]) out3:  torch.Size([16, 17, 2]) integral:  torch.Size([16, 34])

        return [out1,out2,out3], heat_map_intergral.view(-1, self.num_joints, 2)

    def normalize(self, x, heatmap_size=(64, 48)):
        '''x: [N, K, 2]'''
        x[..., 0] = x[..., 0] / heatmap_size[0]
        x[..., 1] = x[..., 1] / heatmap_size[1]

        return x


@GCNS.register_module()
class SemGCN_FC_X(nn.Module):
    
    def __init__(self, adj=None, num_joints=17, hid_dim=None, coords_dim=(2, 2), p_dropout=None):
        super().__init__()

        self.adj = self._build_adj_mx_from_edges(num_joints=num_joints, edge=adj)

        adj = self.adj_matrix
        self.num_joints = num_joints

        self.gconv_input = _GraphConv(adj, coords_dim[0], hid_dim[0], p_dropout=p_dropout)
        # in here we set 4 gcn model in this part
        self.gconv_layers1 = _ResGraphConv_Attention(adj, hid_dim[0], hid_dim[1], hid_dim[0], p_dropout=p_dropout)
        self.gconv_layers2 = _ResGraphConv_Attention(adj, hid_dim[1], hid_dim[2], hid_dim[1], p_dropout=p_dropout)
        self.gconv_layers3 = _ResGraphConv_Attention(adj, hid_dim[2], hid_dim[3], hid_dim[2], p_dropout=p_dropout)
        self.gconv_layers4 = _ResGraphConv_Attention(adj, hid_dim[3], hid_dim[4], hid_dim[3], p_dropout=p_dropout)

        self.gconv_output1 = SemGraphConv(hid_dim[1], coords_dim[1], adj)
        self.gconv_output2 = SemGraphConv(hid_dim[2], coords_dim[1], adj)
        self.gconv_output3 = SemGraphConv(hid_dim[3], coords_dim[1], adj)

    @property
    def adj_matrix(self):
        return self.adj

    @adj_matrix.setter
    def adj_matrix(self,adj_matrix):
        self.adj = adj_matrix

    def _build_adj_mx_from_edges(self,num_joints,edge):
        def adj_mx_from_edges(num_pts, edges, sparse=True):

            edges = np.array(edges, dtype=np.int32)
            data, i, j = np.ones(edges.shape[0]), edges[:, 0], edges[:, 1]
            adj_mx = sp.coo_matrix((data, (i, j)), shape=(num_pts, num_pts), dtype=np.float32)

            # build symmetric adjacency matrix
            adj_mx = adj_mx + adj_mx.T.multiply(adj_mx.T > adj_mx) - adj_mx.multiply(adj_mx.T > adj_mx)
            adj_mx = normalize(adj_mx + sp.eye(adj_mx.shape[0]))
            if sparse:
                adj_mx = sparse_mx_to_torch_sparse_tensor(adj_mx)
            else:
                adj_mx = torch.tensor(adj_mx.todense(), dtype=torch.float)
            return adj_mx

        def sparse_mx_to_torch_sparse_tensor(sparse_mx):
            """Convert a scipy sparse matrix to a torch sparse tensor."""
            sparse_mx = sparse_mx.tocoo().astype(np.float32)
            indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
            values = torch.from_numpy(sparse_mx.data)
            shape = torch.Size(sparse_mx.shape)
            return torch.sparse.FloatTensor(indices, values, shape)

        def normalize(mx):
            """Row-normalize sparse matrix"""
            rowsum = np.array(mx.sum(1))
            r_inv = np.power(rowsum, -1).flatten()
            r_inv[np.isinf(r_inv)] = 0.
            r_mat_inv = sp.diags(r_inv)
            mx = r_mat_inv.dot(mx)
            return mx

        return adj_mx_from_edges(num_joints, edge, False)

    def forward(self, x, ret_features, hm_4=None):
        '''
        TODO x检测的坐标是heatmap上的坐标吧, 回归得到的坐标也是heatmap上面的坐标吧
        x: backbone网络检测的关节点坐标
        hm_4: 输入的用提前检测到的关节点坐标, 如果给了的话就直接用, 不再需要FC来回归出来
        ret_feature: 上采样的不同尺度的featuremap
        '''
        x = torch.from_numpy(x).cuda()
        out = self.gconv_input(x)
        # print('out0: ', out.shape) [16, 17, 128]
        out = self.gconv_layers1(out, None)
        # print('out1: ', out.shape) [16, 17, 128]
        out = self.gconv_layers2(out, None) # [16, 17, 128] [16, 256, 17]
        # print('out1: ', out.shape) # [16, 17, 384]
        out1 = self.gconv_output1(out) # [16, 17, 2]

        out = self.gconv_layers3(out, None) # [16, 17, 384] [16, 256, 17]
        # print('out2: ', out.shape) # [16, 17, 640]
        out2 = self.gconv_output2(out)

        out = self.gconv_layers4(out, None)
        # [16, 17, 768]
        out3 = self.gconv_output3(out)

        return [out1,out2,out3], out3

    def normalize(self, x, heatmap_size=(64, 48)):
        '''x: [N, K, 2]'''
        x[..., 0] = x[..., 0] / heatmap_size[0]
        x[..., 1] = x[..., 1] / heatmap_size[1]

        return x


@GCNS.register_module()
class SemGCN_FC_IM(nn.Module):
    
    def __init__(self, adj=None, num_joints=17, hid_dim=None, coords_dim=(2, 2), p_dropout=None):
        '''
        这里的adj就是人体骨架的连接顺序
        :param adj:  adjacency matrix using for
        :param hid_dim:
        :param coords_dim:
        :param num_layers:
        :param nodes_group:
        :param p_dropout:
        '''
        super().__init__()
        # 这个heatmap_generator是用backbone网络的不同阶段feature生成最后的heatmap, 和不同尺度的feature
        self.heat_map_generator =HM_Extrect(num_joints)

        self.adj = self._build_adj_mx_from_edges(num_joints=num_joints, edge=adj)
        # self.adj_matrix 就是 self.adj
        adj = self.adj_matrix
        self.num_joints = num_joints

        # in here we set 4 gcn model in this part
        self.gconv_layers2 = _ResGraphConv_Attention(adj, 256, 256, 256, p_dropout=p_dropout)
        self.gconv_layers3 = _ResGraphConv_Attention(adj, 256, 256, 256, p_dropout=p_dropout)
        self.gconv_layers4 = _ResGraphConv_Attention(adj, 128, 128, 256, p_dropout=p_dropout)

        self.gconv_output1 = SemGraphConv(256, coords_dim[1], adj)
        self.gconv_output2 = SemGraphConv(256, coords_dim[1], adj)
        self.gconv_output3 = SemGraphConv(128, coords_dim[1], adj)

        #FC
        self.FC = nn.Sequential(nn.Sequential(make_fc(3072,1024),nn.ReLU(inplace=True)),nn.Sequential(make_fc(1024,1024),nn.ReLU(inplace=True)),make_fc(1024,2))

    @property
    def adj_matrix(self):
        return self.adj

    @adj_matrix.setter
    def adj_matrix(self,adj_matrix):
        self.adj = adj_matrix

    def _build_adj_mx_from_edges(self,num_joints,edge):
        def adj_mx_from_edges(num_pts, edges, sparse=True):

            edges = np.array(edges, dtype=np.int32)
            data, i, j = np.ones(edges.shape[0]), edges[:, 0], edges[:, 1]
            adj_mx = sp.coo_matrix((data, (i, j)), shape=(num_pts, num_pts), dtype=np.float32)

            # build symmetric adjacency matrix
            adj_mx = adj_mx + adj_mx.T.multiply(adj_mx.T > adj_mx) - adj_mx.multiply(adj_mx.T > adj_mx)
            adj_mx = normalize(adj_mx + sp.eye(adj_mx.shape[0]))
            if sparse:
                adj_mx = sparse_mx_to_torch_sparse_tensor(adj_mx)
            else:
                adj_mx = torch.tensor(adj_mx.todense(), dtype=torch.float)
            return adj_mx

        def sparse_mx_to_torch_sparse_tensor(sparse_mx):
            """Convert a scipy sparse matrix to a torch sparse tensor."""
            sparse_mx = sparse_mx.tocoo().astype(np.float32)
            indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
            values = torch.from_numpy(sparse_mx.data)
            shape = torch.Size(sparse_mx.shape)
            return torch.sparse.FloatTensor(indices, values, shape)

        def normalize(mx):
            """Row-normalize sparse matrix"""
            rowsum = np.array(mx.sum(1))
            r_inv = np.power(rowsum, -1).flatten()
            r_inv[np.isinf(r_inv)] = 0.
            r_mat_inv = sp.diags(r_inv)
            mx = r_mat_inv.dot(mx)
            return mx

        return adj_mx_from_edges(num_joints, edge, False)

    def forward(self, x, ret_features, hm_4=None):
        '''
        TODO x检测的坐标是heatmap上的坐标吧, 回归得到的坐标也是heatmap上面的坐标吧
        x: backbone网络检测的关节点坐标
        hm_4: 输入的用提前检测到的关节点坐标, 如果给了的话就直接用, 不再需要FC来回归出来
        ret_feature: 上采样的不同尺度的featuremap
        '''
        results, heat_map = self.heat_map_generator(ret_features)
        bs = heat_map.shape[0]
        # TODO 积分英文 `integral`
        heat_map_intergral = self.FC(heat_map.view(bs*self.num_joints, -1)).view(bs, self.num_joints*2)
        # print('heatmap_integral: ', heat_map_intergral.shape) # [N, 34]

        hm_4 = heat_map_intergral.view(-1, self.num_joints, 2)
        j_1_16 = F.grid_sample(results[0],hm_4[:,None,:,:], align_corners=False).squeeze(2).permute(0, 2, 1)
        j_1_8 = F.grid_sample(results[1],hm_4[:,None,:,:], align_corners=False).squeeze(2).permute(0, 2, 1)
        j_1_4 = F.grid_sample(results[2],hm_4[:,None,:,:], align_corners=False).squeeze(2).permute(0, 2, 1)

        out = self.gconv_layers2(j_1_16, None) # [16, 17, 128] [16, 256, 17]
        out1 = self.gconv_output1(out) # [16, 17, 2]

        out = self.gconv_layers3(j_1_8, None) # [16, 17, 384] [16, 256, 17]
        out2 = self.gconv_output2(out)

        out = self.gconv_layers4(j_1_4, None)
        out3 = self.gconv_output3(out)

        return [out1,out2,out3], heat_map_intergral.view(-1, self.num_joints, 2)

    def normalize(self, x, heatmap_size=(64, 48)):
        '''x: [N, K, 2]'''
        x[..., 0] = x[..., 0] / heatmap_size[0]
        x[..., 1] = x[..., 1] / heatmap_size[1]

        return x


class HM_Extrect(nn.Module):

    def __init__(self,out_channel):
        super(HM_Extrect,self).__init__()


        self.level_conv1_1 = make_conv3x3(256,256)
        self.level_conv2_1 = make_conv3x3(256,256)
        self.level_conv3_1 = make_conv3x3(256,128)
        self.level_conv_out = make_conv3x3(128, out_channel)
        self.level_conv1_up = nn.ConvTranspose2d(256, 256, 2, 2, 0)
        self.level_conv2_up = nn.ConvTranspose2d(256, 256, 2, 2, 0)
        self.attention_conv1_up_a = Attention_layer(512,256,reduction = 1)
        self.attention_conv2_up_a = Attention_layer(512,256,reduction = 1)

    def forward(self,features):
        results = []

        x = F.relu(self.level_conv1_1(features[0])) # [256]
        results.append(x)
        x = F.relu(self.level_conv1_up(x))          # [256]
        x = self.attention_conv1_up_a(x,features[1])
        x = (F.relu(self.level_conv2_1(x)))         # [256]
        results.append(x)

        x = F.relu(self.level_conv2_up(x))          # [256]
        # print('x: ', x.shape, ' features[2]: ', features[2].shape)
        x = self.attention_conv2_up_a(x,features[2])
        x = (F.relu(self.level_conv3_1(x)))
        results.append(x)
        return results,self.level_conv_out(x)