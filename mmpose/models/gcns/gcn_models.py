import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class BaseGCN(nn.Module):
    """
    base graph convolution layer
    """
    def __init__(self, in_channels=2, out_channels=2, adj=None, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.W = nn.Linear(2, 2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 2)
        self.relu2 = nn.ReLU()

        self.adj = adj.cuda()

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels, dtype=torch.float))
            stdv = 1. / math.sqrt(self.adj.size(-1))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        """ x: N, num_joints, 2
            return: N*
        """
        N, J = x.shape[:2]
        x = x.view(N*J, -1)
        res = self.fc2(x)

        x = self.W(x)
        x = torch.bmm(self.adj.expand(N, J, J), x.view(N, J, -1))
        x = self.bn(x.permute(0, 2, 1))
        x = self.relu1(x)
        # print(x.shape) # N, 2, J
        x = self.fc1(x.permute(0, 2, 1).reshape(N*J, -1))

        x = self.relu2(x + res)

        return x.view(N, J, -1)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_channels) + ' -> ' + str(self.out_channels) + ')'