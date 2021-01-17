import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse as sp
import numpy as np
import math


class _GraphConv(nn.Module):
    def __init__(self, adj, input_dim=16, output_dim=16, num_joints=17, p_dropout=None):
        super(_GraphConv, self).__init__()

        self.gconv = PoseGraphConv(input_dim, output_dim, adj)
        self.bn = nn.BatchNorm2d(output_dim*num_joints)
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
    pose graph convolution layer
    """

    def __init__(self, in_features=16, out_features=16, adj=None, num_joints=17):
        super(PoseGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_joints = num_joints
        
        # use group convolution to implement transformation matrix in paper
        self.W = nn.Conv2d(
            in_channels=num_joints * out_features, 
            out_channels=num_joints * out_features,
            kernel_size=3, 
            padding=1,
            groups=num_joints
            )

        self.adj = adj

        self.conv_att = nn.Sequential(
            nn.Conv2d(
                in_channels=out_features*2, 
                out_channels=1, 
                kernel_size=1
            ),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        '''
        backbone: [N, J, H, W] ->conv1d-> [N, J*C, H, W]
        input: [N, J*C, H, W]
        return: [N, J*C, H, W]
        '''
        adj = F.softmax(self.adj, dim=1)

        J, C = self.num_joints, self.out_features
        N, JC, H, W = input.shape
        assert JC == J*C
        # [N, JC, H, W]
        B = self.W(input)

        # L-PGCN spatial attention
        B_list = torch.split(B, C, dim=1)
        S = []
        for i in range(J):
            # print(B_list[0].shape) [N, C, H, W]
            # expand b_uv to [N, J, H, W, d] and concat, then apply conv to get spatial attention
            # [N, C, H, W] -> [N, J, C, H, W] -> [N, J, 2C, H, W] -> [N*J, 1, H, W]
            s_uv = self.conv_att(
                torch.cat(
                    (B_list[i].unsqueeze(1).expand(N, J, C, H, W), B.reshape(N, J, C, H, W)), dim=2
                    ).reshape(N*J, 2*C, H, W)
                    )
            s_uv = s_uv.reshape(N, J, 1, H, W)
            # s_uv [N, J, 1, H, W]
            S.append(s_uv)

        # Z_u
        Z = []
        for i in range(J):
            # [N, J, C, H, W] * [N, J, 1, H, W]
            b_uv = B.reshape(N, J, C, H, W) * S[i]
            # sum
            # [1, J] [N, J, C, H, W] -> [N, 1, J] [N, J, C*H*W] -> [N, 1, C, H, W]
            z_u = torch.bmm(adj[i].expand(N, 1, J), b_uv.reshape(N, J, C*H*W))
            Z.append(z_u.reshape(N, C, H, W))

        return torch.cat(Z, dim=1)


    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class PGCN(nn.Module):
    def __init__(self, in_channels=16, out_joints=17, num_layer=2):
        super(PGCN, self).__init__()
        
        edge = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], \
            [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]]
        self.adj = self._build_adj_mx_from_edges(num_joints=17, edge=edge)
        self.L_pgcn1 = _GraphConv(adj=self.adj)
        self.L_pgcn2 = _GraphConv(adj=self.adj)

        self.final_layer = nn.Conv2d(
            in_channels=in_channels*out_joints, 
            out_channels=out_joints,
            kernel_size=1
        )

    def forward(self, x):
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
            print(adj_mx)
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

if __name__ == "__main__":
    net = PGCN()
    import numpy as np
    param_num = 0
    for tag, val in net.named_parameters():
        param_num += np.prod(val.shape)
    print('param: ', param_num)

    input = torch.rand(8, 16*17, 64, 48)
    out = net(input)
    print(out.shape)

# class PoseGraphConv_bak(nn.Module):
#     """
#     pose graph convolution layer
#     """

#     def __init__(self, in_features=16, out_features=16, adj=None, bias=True):
#         super(PoseGraphConv, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
        
#         #very useful demo means this is Parameter, which can be adjust by bp methods
#         self.W = nn.Parameter(torch.zeros(size=(17, in_features, out_features), dtype=torch.float))
#         nn.init.xavier_uniform_(self.W.data, gain=1.414)

#         self.adj = adj

#         self.conv_att = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=32, 
#                 out_channels=1, 
#                 kernel_size=1
#             ),
#             nn.ReLU(inplace=True)
#         )

#         # if bias:
#         #     self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
#         #     stdv = 1. / math.sqrt(self.W.size(2))
#         #     self.bias.data.uniform_(-stdv, stdv)
#         # else:
#         #     self.register_parameter('bias', None)

#     def forward(self, input):
#         '''
#         input: [N, j, h, w, d]
#         '''
#         adj = F.softmax(self.adj, dim=1)

#         N, J, H, W, d = input.shape
#         b_list = []
#         for i in range(J):
#             b_u = input[:, i, ...].reshape(N*H*W, d)
#             b_u = torch.matmul(b_u, self.W[node]).reshape(N, H, W, d).unsqueeze(1)
#             b_list.append(b_u)

#         # [N, J, H, W, d]
#         b = torch.stack(b_list, dim=1)

#         # L-PGCN spatial attention
#         s = []
#         for i in range(J):
#             # expand b_uv to [N, J, H, W, d] and concat, then apply conv to get spatial attention
#             # [N*J, 1, H, W]
#             s_uv = self.conv_att(torch.cat((b_list[i].expand(N, J, H, W, d), b), dim=4).permute(0, 1, 4, 2, 3).reshape(N*J, 1, H, W))
#             # [N, J, H, W, 1]
#             s_uv = s_uv.permute(0, 2, 3, 1).reshape(N, J, H, W, 1)
#             # att_i [N, J, H, W, 1]
#             att.append(s_uv)


#         # Z_u
#         z = []
#         for i in range(J):
#             # [N, J, H, W, d]
#             b_uv = b * att[i]
#             # sum
#             # [N, J, H, W, d] [J, 1]
#             z_u = torch.bmm(adj[i].expand(N, 1, J), b_uv.reshape(N, J, H*W*d)).reshape(N, 1, H, W, d)
#             z.append(z_u)

#         return torch.stack(z, dim=1)


#     def __repr__(self):
#         return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'