import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse as sp
import numpy as np
import math


class _GraphConv(nn.Module):
    def __init__(self, adj, in_channels=1, out_channels=1, num_joints=17, p_dropout=None, type='att'):
        super(_GraphConv, self).__init__()

        # self.gconv = PoseGraphConv(input_dim, output_dim, adj)
        if type == 'att':
            self.gconv = PoseGraphConv(in_channels, out_channels, adj)
            self.bn = nn.BatchNorm2d(out_channels*num_joints)
        elif type == 'non_local':
            self.gconv = NonLocalGraphConv(in_channels=in_channels, inter_channels=out_channels, adj=adj)
            self.bn = nn.BatchNorm2d(in_channels*num_joints)
        self.relu = nn.ReLU()

        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = None

    def forward(self, x):
        #in here if our batch size equal to 64
        # x[N, J, H, W, d]

        x = self.gconv(x)
        # print(x.shape)
        x = self.bn(x)
        if self.dropout is not None:
            x = self.dropout(self.relu(x))

        x = self.relu(x)
        return x


class PoseGraphConv(nn.Module):
    """
    local pose graph convolution block
    """
    def __init__(self, in_channels=16, out_channels=16, adj=None, num_joints=17):
        super(PoseGraphConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_joints = num_joints
        
        # use group convolution to implement transformation matrix in paper
        # self.W = nn.Parameter(torch.zeros(size=(in_features, out_features), dtype=torch.float))
        self.W = nn.Conv2d(
            in_channels=num_joints * in_channels, 
            out_channels=num_joints * out_channels,
            kernel_size=3, 
            padding=1,
            groups=num_joints
            )

        self.adj = adj.cuda()

        self.conv_att = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels*2, 
                out_channels=1, 
                kernel_size=1
            ),
            nn.Sigmoid(),
        )

        self.att = True

    def forward(self, input):
        '''
        backbone: [N, J, H, W] ->conv1d-> [N, J*C, H, W]
        input: [N, J*C, H, W]
        return: [N, J*C, H, W]
        '''
        adj = F.softmax(self.adj, dim=1)

        J, C = self.num_joints, self.in_channels
        # print(J, C)
        N, JC, H, W = input.shape
        assert JC == J*C
        # [N, JC, H, W]
        B = self.W(input)
        C = self.out_channels

        # L-PGCN spatial attention
        if self.att:
            # [N, J*C, H, W] -> [N, C, H, W] * J
            B_list = torch.split(B, C, dim=1)
            S = []
            for i in range(J):
                # expand b_uv to [N, J, H, W, d] and concat, then apply conv to get spatial attention
                # [N, C, H, W] --expand--> [N, J, C, H, W] --cat--> [N, J, 2C, H, W] -> [N*J, 1, H, W]
                s_uv = self.conv_att(
                    torch.cat(
                        (B_list[i].unsqueeze(1).expand(N, J, C, H, W), B.reshape(N, J, C, H, W)), dim=2
                        ).view(N*J, 2*C, H, W))
                # s_uv [N*J, 1, H, W]
                S.append(s_uv)

            # Z_u
            Z = []
            for i in range(J):
                # [N, J, C, H, W] * [N, J, 1, H, W]
                b_uv = B.view(N*J, C, H, W) * S[i]
                # sum
                # [1, J] [N, J, C, H, W] -> [N, 1, J] [N, J, C*H*W] -> [N, 1, C, H, W]
                z_u = torch.bmm(adj[i].expand(N, 1, J), b_uv.reshape(N, J, C*H*W))
                Z.append(z_u.reshape(N, C, H, W))
        
            return torch.cat(Z, dim=1)

        else:
            # adj: [J, J] X [N, JC, H, W]
            Z = torch.bmm(adj.expand(N, J, J), B.reshape(N, J, C*H*W))

            return Z.reshape(N, J*C, H, W)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_channels) + ' -> ' + str(self.out_channels) + ')'

class NonLocalGraphConv(nn.Module):
    """
    non-local pose graph convolution block
    """

    def __init__(self, in_channels=1, inter_channels=4, adj=None, num_joints=17, bn_layer=True):
        super(NonLocalGraphConv, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.num_joints = num_joints
        self.adj=adj

        # use group convolution to implement transformation matrix in paper
        # 
        self.g = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=inter_channels, 
            kernel_size=3, 
            stride=1,
            padding=1
        )

        self.theta = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=inter_channels, 
            kernel_size=3, 
            stride=1,
            padding=1
        )

        self.phi = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=inter_channels, 
            kernel_size=3, 
            stride=1,
            padding=1
        )

        if bn_layer:
            self.W = nn.Sequential(
                nn.Conv2d(
                    in_channels=inter_channels, 
                    out_channels=in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                ),
                nn.BatchNorm2d(in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = nn.Conv2d(
                in_channels=inter_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=1,
                padding=1
            )
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

    def forward(self, input, return_nl_map=False):
        '''
        backbone: [N, J, H, W] ->conv1d-> [N, J*C, H, W]
        input: [N, J*C, H, W]
        return: [N, J*C, H, W]
        '''
        # print(input.shape)
        adj = F.softmax(self.adj.cuda(), dim=1)

        J, C = self.num_joints, self.in_channels
        # print(self.in_channels)
        # print(J, C)
        N, JC, H, W = input.shape
        assert JC == J*C
        # [N, JC, H, W] -> [N*J, C, H, W]
        input = input.view(N*J, C, H, W)
        g_x = self.g(input).view(N*J, self.inter_channels, H*W).permute(0, 2, 1)
        # print(g_x.shape)
        theta_x = self.theta(input).view(N*J, self.inter_channels, H*W).permute(0, 2, 1)
        # print(theta_x.shape)
        phi_x = self.phi(input).view(N*J, self.inter_channels, H*W)
        # print(phi_x.shape)
        # [N*J, H*W, C], [N*J, C, H*W] -> [N*J, HW, HW]
        # f_div_C = F.softmax(torch.matmul(theta_x, phi_x), dim=-1)
        f_div_C = torch.matmul(theta_x, phi_x) / H*W
        # [N*J, HW, HW] * [N*J, HW, C] -> [N*J, HW, C] -> [N*J, C, HW] -> [N*J, C, H, W]
        y = torch.matmul(f_div_C, g_x).permute(0, 2, 1).reshape(N*J, self.inter_channels, H, W)

        W_y = self.W(y)
        Z = W_y + input

        # Z = torch.bmm(adj.expand(N, J, J), Z.reshape(N, J, C*H*W))

        if return_nl_map:
            return Z.reshape(N, J*C, H, W), f_div_C
        
        return Z.reshape(N, J*C, H, W)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_channels) + ' -> ' + str(self.in_channels) + ')'


class PGCN(nn.Module):
    """Implementation of PGCN proposed in ``Structure-aware human pose 
    estimation with graph convolutional networks``.

    Args:
        in_channels (int): Number of input channels of _GraphConv.
        out_channels (int): Number of output channels or inter_channels of non-local 
        _GraphConv if type is 'non-local'.
        out_joints (int): Number of keypoint joints.
        type (str): Type of _GraphConv, `att`: L-PGCN, `non-local`: non-local PGCN, 
        `combined`(TODO): full version of PGCN, which consists of L-PGCN and non-local PGCN
    """
    def __init__(self, in_channels=1, out_channels=16, out_joints=17, type='att'):
        super(PGCN, self).__init__()
        '''if type is 'non-local', out_channels means inter_channels of non-local block'''

        edge = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], \
            [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]]
        self.adj = self._build_adj_mx_from_edges(num_joints=17, edge=edge)

        self.L_pgcn1 = _GraphConv(
            in_channels=in_channels, 
            out_channels=out_channels, 
            adj=self.adj, 
            type=type)
        if type == 'att':
            self.L_pgcn2 = _GraphConv(
                in_channels=out_channels, 
                out_channels=out_channels, 
                adj=self.adj, 
                type=type)
        else:
            self.L_pgcn2 = _GraphConv(
                in_channels=in_channels, 
                out_channels=out_channels, 
                adj=self.adj, 
                type=type)

        in_channels_final = out_channels*out_joints if type == 'att' else in_channels*out_joints
        self.final_layer = nn.Conv2d(
            in_channels=in_channels*out_joints, 
            out_channels=out_joints,
            kernel_size=1
        )

    def forward(self, x):
        # print(x.shape)
        x = self.L_pgcn1(x)
        x = self.L_pgcn2(x)
        x = self.final_layer(x)

        return x

    def _build_adj_mx_from_edges(self,num_joints,edge):
        def adj_mx_from_edges(num_pts, edges, sparse=True):
            # print(num_pts)
            # print(edges)
            
            edges = np.array(edges, dtype=np.int32)
            # print(edges)
            data, i, j = np.ones(edges.shape[0]), edges[:, 0], edges[:, 1]
            adj_mx = sp.coo_matrix((data, (i, j)), shape=(num_pts, num_pts), dtype=np.float32)

            # build symmetric adjacency matrix
            adj_mx = adj_mx + adj_mx.T.multiply(adj_mx.T > adj_mx) - adj_mx.multiply(adj_mx.T > adj_mx)
            adj_mx = normalize(adj_mx + sp.eye(adj_mx.shape[0]))
            if sparse:
                adj_mx = sparse_mx_to_torch_sparse_tensor(adj_mx)
            else:
                adj_mx = torch.tensor(adj_mx.todense(), dtype=torch.float)
            # print(adj_mx)
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

# if __name__ == "__main__":
#     net = PGCN(type='non_local').cuda()
#     import numpy as np
#     param_num = 0
#     for tag, val in net.named_parameters():
#         param_num += np.prod(val.shape)
#     print('param: ', param_num)

#     input = torch.rand(8, 17, 64, 48)
#     out = net(input)
#     print(out.shape)
