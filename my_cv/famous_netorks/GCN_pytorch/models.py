from __future__ import division
from __future__ import print_function

import argparse
import math

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.nn.parameter import Parameter


class GCN(nn.Module):
    # dropout_rate = 0.2

    def __init__(self, n_in, n_out, dropout_rate=None):
        super(GCN, self).__init__()
        self.graph1 = GraphConvolution(n_in, n_out)
        self.dropout_rate = dropout_rate

    def forward(self, x, adj):
        # todo 不知道adj的意义
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


def encode_onehot(labels):
    """onehot编码，将每个类别转成一个向量"""
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


# def load_data(path="../data/cora/", dataset="cora"):
def load_data(path="sample_data/cora/", dataset="cora"):
    """
    加载数据
    :param path: 数据的路径
    :param dataset: 数据集名
    :return:
    """
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))
    # 解析数据
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    # 从第2列到倒数第二列是特征数据
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # 最后一列是标签数据
    labels = encode_onehot(idx_features_labels[:, -1])

    # 索引数据
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # 保存数据的索引，key是数据id，value是原数据中的位置（从0开始）。
    idx_map = {j: i for i, j in enumerate(idx)}
    # 加载数据
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    # 将数据压平,根据id找到对应的索引的位置
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # 正规化特征数据
    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    # 将邻接矩阵转换为稀疏矩阵
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


# 加载数据
adj, features, labels, idx_train, idx_val, idx_test = load_data()

# 建立模型
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)
# GCN的损失函数分为两部分，一部分是分类损失，一部分是权重的正则化。
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    # 将特征和邻接矩阵输入网络，得到输出的数据
    output = model(features, adj)
    # 计算损失
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    # 计算正确率
    acc_train = accuracy(output[idx_train], labels[idx_train])
    # 反向传播
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # 进入评估模式
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    """测试数据"""
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


import time

t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()
