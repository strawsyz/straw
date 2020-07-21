#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2020/7/21 14:36
# @Author  : Shi
# @FileName: CNN1D.py
# @Description：

from torch import nn

"""使用一维的卷积组成的网络"""


class CNN1DBlock(nn.Module):
    def __init__(self, in_chans, out_chans, k_size=2, pool_size=2):
        super(CNN1DBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(in_chans, out_chans, k_size),
            nn.BatchNorm1d(out_chans),
            nn.ReLU(),
            nn.MaxPool1d(pool_size)
        )

    def forward(self, x):
        out = self.layers(x)
        return out


class CNN1DBlockNoPool(nn.Module):
    def __init__(self, in_chans, out_chans, k_size=2):
        super(CNN1DBlockNoPool, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(in_chans, out_chans, k_size),
            nn.BatchNorm1d(out_chans),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.layers(x)
        return out


class CNN1D(nn.Module):
    def __init__(self):
        super(CNN1D, self).__init__()
        factor = 4
        self.cnn1_layers = []
        self.cnn2_layers = []
        in_chans = 1
        last_chans = in_chans
        for i in range(5):
            cnn1d_block = CNN1DBlock(in_chans=last_chans, out_chans=last_chans * factor, k_size=2, pool_size=factor)
            setattr(self, "cnn1d_blocks_1_{}".format(i), cnn1d_block)
            last_chans *= factor
        self.pool1 = nn.MaxPool1d(4)
        for i in range(5):
            cnn1d_block = CNN1DBlockNoPool(in_chans=last_chans, out_chans=int(last_chans / factor), k_size=1)
            setattr(self, "cnn1d_blocks_2_{}".format(i), cnn1d_block)
            last_chans = int(last_chans / factor)

    def forward(self, x):
        out = x
        for layer in self.cnn1_layers:
            out = layer(out)
        for layer_no in range(5):
            out = getattr(self, "cnn1d_blocks_1_{}".format(layer_no))(out)
        for layer in self.cnn2_layers:
            out = getattr(self, "cnn1d_blocks_2_{}".format(layer_no))(out)
            # out = layer(out)
        out = self.pool1(out)

        return out


class MyCNN1D(nn.Module):

    def __init__(self):
        super(MyCNN1D, self).__init__()
        for i in range(8):
            setattr(self, "cnn1d_{}".format(i), CNN1D())

    def forward(self, input):
        input = input.reshape(8, 5000)
        cnn_layer = getattr(self, "cnn1d_0")
        output = cnn_layer(input[0][None, None, :])
        for i in range(1, 8):
            cnn_layer = getattr(self, "cnn1d_{}".format(i))
            output += cnn_layer(input[i][None, None, :])
        return output


if __name__ == '__main__':
    import torch

    data = torch.rand((1, 40000))
    net = MyCNN1D()
    print(net.parameters())
    out = net(data)
    print(out)
    print(net.parameters())
