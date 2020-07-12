import math

import torch
from torch import functional as F
from torch import nn
from torch.nn.parameter import Parameter


class GCN(nn.Module):
    # dropout_rate = 0.2

    def __init__(self, n_in, n_out, dropout_rate=None):
        super(GCN, self).__init__()
        self.graph1 = GraphConvolution(n_in, n_out)
        self.dropout_rate = dropout_rate

    def forward(self, x, adj):
        # adj是邻接矩阵
        out = F.relu(self.graph1(x, adj), inplace=True)
        if self.dropout_rate is not None:
            return F.dropout(out, self.dropout_rate, training=self.training)
        else:
            return out


class GraphConvolution(nn.Module):
    """
    使用pytorch实现的图卷积层
    """

    def __init__(self, n_in, n_out, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = n_in
        self.out_features = n_out
        # 建立边权重
        self.weight = Parameter(torch.FloatTensor(n_in, n_out))
        if bias:
            self.bias = Parameter(torch.FloatTensor(n_out))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        """
        前向传播
        :param input: 输入数据
        :param adj: 邻接矩阵
        :return:
        """
        # 两个矩阵相乘
        support = torch.mm(input, self.weight)
        # 左乘标准化的邻接矩阵
        # 于邻接矩阵的存储时用的是稀疏矩阵，所以有别于上一行
        # torch.spmm表示sparse_tensor与dense_tensor相乘。
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
