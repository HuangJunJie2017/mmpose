import torch.nn as nn
import torch
from .semgcn_utils import *
from .semgcn import _ResGraphConv_Attention,SemGraphConv,_GraphConv
from scipy import sparse as sp
import numpy as np

from mmcv.cnn import (build_conv_layer, build_upsample_layer, constant_init,
                      normal_init)
from .gcn_models import BaseGCN
# import registor
from ..registry import GCNS


@GCNS.register_module()
class BaseSSRNet(nn.Module):
    """
    Args:
        in_channels (list): Number of input channels of 
            the backbone features fed into the head layer.
        adj (list): list of skeleton connections.
        num_joints (int): Number of joints.
        coords_dim (tuple): Dimension of the joint coords.
        IS (bool): Whether use intermediate supervision.
        p (float): value of lambda in paper.
    """
    def __init__(self, 
                 in_channels=(256, 256, 256),
                 num_stage=3,
                 num_deconv_filters=(256, 256, 256), 
                 num_deconv_kernels=(4, 4, 4),
                 adj=None,
                 hid_dim=None,
                 num_joints=17, 
                 coords_dim=(2, 2), 
                 IS=False, 
                 p=0.5):
        super().__init__()

        self.in_channels = in_channels
        self.adj = self._build_adj_mx_from_edges(num_joints=num_joints, edge=adj)
        self.num_joints = num_joints
        self.IS = IS
        self.p = p

        self.head0, self.head1, self.head2 = self._make_head_layer(num_stage, num_deconv_filters, num_deconv_kernels)
        self.encoder, self.decoder = self._make_encoder_decoder_layer()

        self.gcn0 = BaseGCN(adj=self.adj)
        self.gcn1 = BaseGCN(adj=self.adj)
        self.gcn2 = BaseGCN(adj=self.adj)

    def forward(self, features):
        '''
        features: 上采样的不同尺度的featuremap
        '''
        N = features[0].shape[0]
        hm0 = self.head0(features[0]) # N, J, H, W
        hm1 = self.head1(features[1])
        hm2 = self.head2(features[2])

        # print('hm0: ', hm0.shape)

        c0 = self.decoder(hm0.view(N * self.num_joints, -1)).view(N, self.num_joints, -1) # N, J, 2
        # print('c0: ', c0.shape)
        c0_g = self.gcn0(c0)
        c1 = self.decoder(hm1.view(N * self.num_joints, -1)).view(N, self.num_joints, -1)
        c1_g = self.gcn1(self.p * c1 + (1 - self.p) * (c0 + c0_g))
        c2 = self.decoder(hm2.view(N * self.num_joints, -1)).view(N, self.num_joints, -1)
        c2_g = self.gcn2(self.p * c2 + (1 - self.p) * (c1 + c1_g))

        if self.IS:
            return [hm0, hm1, hm2], [c0_g+c0, c1_g+c1, c2_g+c2]
        else:
            return [hm2], [c2_g+c2]

    def _make_head_layer(self, num_layers, num_filters, num_kernels):
        """Make head layers."""
        if num_layers != len(num_filters):
            error_msg = f'num_layers({num_layers}) ' \
                        f'!= length of num_filters({len(num_filters)})'
            raise ValueError(error_msg)
        if num_layers != len(num_kernels):
            error_msg = f'num_layers({num_layers}) ' \
                        f'!= length of num_kernels({len(num_kernels)})'
            raise ValueError(error_msg)
        if num_layers != len(self.in_channels):
            error_msg = f'num_layers({num_layers}) ' \
                        f'!= length of num_filters({len(self.in_channels)})'
            raise ValueError(error_msg)

        layers = []
        for i in range(num_layers):
            in_channels = self.in_channels[i]
            scale_factor = 2 ** (num_layers - 1 - i)
            layer = []

            planes = num_filters[i]
            for j in range(num_layers-1-i):
                kernel, padding, output_padding = self._get_deconv_cfg(num_kernels[i])
                layer.append(
                    build_upsample_layer(
                        dict(type='deconv'),
                        in_channels=in_channels,
                        out_channels=planes,
                        kernel_size=kernel,
                        stride=2,
                        padding=padding,
                        output_padding=output_padding,
                        bias=False)
                )
                layer.append(nn.BatchNorm2d(planes))
                layer.append(nn.ReLU(inplace=True))
                in_channels = planes

            layer.append(
                build_conv_layer(
                    cfg=dict(type='Conv2d'),
                    in_channels=num_filters[-1],
                    out_channels=self.num_joints,
                    kernel_size=1,
                    stride=1,
                    padding=0))
            layers.append(nn.Sequential(*layer))
        
        return layers

    def _make_encoder_decoder_layer(self, out_channels=128, kernel=3, size=(64, 48)):
        layers = []
        layers.append(nn.Sequential(
            nn.Conv2d(in_channels=self.num_joints, 
                      out_channels=out_channels // 4 * self.num_joints, 
                      kernel_size=3, 
                      padding=1, 
                      stride=2, 
                      groups=self.num_joints), 
            nn.BatchNorm2d(num_features=out_channels // 4 * self.num_joints),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels // 4 * self.num_joints, 
                      out_channels=out_channels * self.num_joints, 
                      kernel_size=3, 
                      padding=1, 
                      stride=2, 
                      groups=self.num_joints),
            nn.BatchNorm2d(num_features=out_channels * self.num_joints), 
            nn.ReLU(), 
            nn.AdaptiveAvgPool2d(1)
        ))

        layers.append(nn.Sequential(
            nn.Linear(size[0] * size[1], 1024), 
            nn.ReLU(inplace=True), 
            nn.Linear(1024, 256), 
            nn.ReLU(), 
            nn.Linear(256, 2)))
        
        return layers

    @staticmethod
    def _get_deconv_cfg(deconv_kernel):
        """Get configurations for deconv layers."""
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise ValueError(f'Not supported num_kernels ({deconv_kernel}).')

        return deconv_kernel, padding, output_padding

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