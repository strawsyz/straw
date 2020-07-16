#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2020/7/15 13:51
# @Author  : Shi
# @FileName: BaseFNN.py
# @Description：
from torch import nn


class LinearReLU(nn.Sequential):
    def __init__(self, in_n, out_n):
        super(LinearReLU, self).__init__()
        self.add_module("Linear", nn.Linear(in_n, out_n))
        self.add_module("ReLU", nn.ReLU(inplace=True))


class LinearReLUDropout(nn.Sequential):
    def __init__(self, in_n, out_n, dropout_rate=0.3):
        super(LinearReLUDropout, self).__init__()
        self.add_module("Linear", nn.Linear(in_n, out_n))
        self.add_module("ReLU", nn.ReLU(inplace=True))
        self.add_module("Dropout", nn.Dropout(dropout_rate))


class DenseLayer(nn.Sequential):
    def __init__(self, n_in=500, n_out=3, n_hides=[256, 128, 64, 32, 16], dropout=None):
        super(DenseLayer, self).__init__()
        for i, n_hide in enumerate(n_hides):
            if dropout is not None:
                self.add_module("LinearReLU_{}".format(i), LinearReLUDropout(n_in, n_hide, dropout_rate=dropout))
            else:
                self.add_module("LinearReLU_{}".format(i), LinearReLU(n_in, n_hide))
            nn.Linear(n_in, n_hide)
            n_in = n_hide
        self.add_module("Classier", nn.Linear(n_in, n_out))


class MyFNN(nn.Module):
    def __init__(self, n_out=3, dropout=None):
        super(MyFNN, self).__init__()
        self.linears = []
        for i in range(8):
            from operator import attrgetter
            attr = attrgetter("linear{}".format(i))
            setattr(self, "linear{}".format(i),
                    DenseLayer(n_in=5000, n_out=n_out, n_hides=[2048, 1024, 512, 256, 128, 64, 32, 16],
                               dropout=dropout))
            self.linears.append(attr(self))

    def forward(self, input):
        input = input.reshape(8, 5000)
        # print(input[0].shape)
        output = self.linears[0](input[0])
        for i in range(1, 8):
            output += self.linears[i](input[i])
        return output
