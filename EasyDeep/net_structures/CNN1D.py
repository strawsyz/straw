#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2020/7/21 14:36
# @Author  : Shi
# @FileName: CNN1D.py
# @Description：

from torch import nn
import torch
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
        for layer_no in range(5):
            out = getattr(self, "cnn1d_blocks_2_{}".format(layer_no))(out)
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


class MultiLabelCNN1D(nn.Module):
    def __init__(self):
        super(MultiLabelCNN1D, self).__init__()
        self.conv_1 = CNN1DBlockNoPool(1, 4)
        self.conv_2 = CNN1DBlockNoPool(4, 16, k_size=8)
        self.conv_3 = CNN1DBlockNoPool(16, 16, k_size=8)
        self.conv_4 = CNN1DBlockNoPool(16, 32, k_size=8)
        self.conv_5 = CNN1DBlockNoPool(32, 64, k_size=8)
        self.conv_6 = CNN1DBlockNoPool(64, 32, k_size=8)
        self.conv_7 = CNN1DBlockNoPool(32, 16, k_size=8)
        self.conv_8 = CNN1DBlockNoPool(16, 11, k_size=3)

    def forward(self, input):
        input = torch.unsqueeze(input, dim=1)
        output = self.conv_1(input)
        output = self.conv_2(output)
        output = self.conv_3(output)
        output = self.conv_4(output)
        output = self.conv_5(output)
        output = self.conv_6(output)
        output = self.conv_7(output)
        output = self.conv_8(output)
        output = torch.squeeze(output,dim=2)
        return output


if __name__ == '__main__':

    data = torch.rand((4, 1, 46))
    net = MultiLabelCNN1D()
    out = net(data)
    print(out.shape)
