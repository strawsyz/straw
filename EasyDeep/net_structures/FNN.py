#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2020/7/15 13:51
# @Author  : Shi
# @FileName: FNN.py
# @Description：
from torch import nn
import torch


class Linear(nn.Sequential):

    def __init__(self, num_in, num_out, init=False):
        super(Linear, self).__init__()
        self.add_module("Linear", nn.Linear(num_in, num_out))
        if init:
            self.init()

    def init(self):
        torch.nn.init.xavier_uniform_(self.Linear.weight)
        torch.nn.init.constant_(self.Linear.bias, 0)


class LinearReLU(nn.Sequential):
    def __init__(self, num_in, num_out, init=False):
        super(LinearReLU, self).__init__()
        self.add_module("Linear", nn.Linear(num_in, num_out))
        self.add_module("ReLU", nn.ReLU(inplace=True))
        if init:
            self.init()

    def init(self):
        torch.nn.init.xavier_uniform_(self.Linear.weight)
        torch.nn.init.constant_(self.Linear.bias, 0)


class LinearSigmoid(nn.Sequential):
    def __init__(self, num_in, num_out, init=False):
        super(LinearSigmoid, self).__init__()
        self.add_module("Linear", nn.Linear(num_in, num_out))
        self.add_module("Sigmoid", nn.Sigmoid())
        if init:
            self.init()

    def init(self):
        torch.nn.init.xavier_uniform_(self.Linear.weight)
        torch.nn.init.constant_(self.Linear.bias, 0)


class LinearReLUDropout(nn.Sequential):
    def __init__(self, num_in, num_out, dropout_rate=0.3):
        super(LinearReLUDropout, self).__init__()
        self.add_module("Linear", nn.Linear(num_in, num_out))
        self.add_module("ReLU", nn.ReLU(inplace=True))
        self.add_module("Dropout", nn.Dropout(dropout_rate))


class DenseLayer(nn.Sequential):
    def __init__(self, num_in=500, num_out=3, n_hides=[256, 128, 64, 32, 16], dropout=None):
        super(DenseLayer, self).__init__()
        for i, n_hide in enumerate(n_hides):
            if dropout is not None:
                self.add_module("LinearReLU_{}".format(i), LinearReLUDropout(num_in, n_hide, dropout_rate=dropout))
            else:
                self.add_module("LinearReLU_{}".format(i), LinearReLU(num_in, n_hide))
            nn.Linear(num_in, n_hide)
            num_in = n_hide
        self.add_module("Classier", nn.Linear(num_in, num_out))


class FNNWithDropout(nn.Module):
    def __init__(self, n_in, n_out, dropout=0.05):
        super(FNNWithDropout, self).__init__()
        self.linears = []
        for i in range(8):
            from operator import attrgetter
            attr = attrgetter("linear{}".format(i))
            setattr(self, "linear{}".format(i),
                    DenseLayer(num_in=n_in, num_out=n_out, n_hides=[2048, 1024, 512, 256, 128, 64, 32, 16],
                               dropout=dropout))
            self.linears.append(attr(self))

    def forward(self, input):
        input = input.reshape(8, -1)
        output = self.linears[0](input[0])
        for i in range(1, 8):
            output += self.linears[i](input[i])
        return output


class AdaptiveFNN(nn.Module):
    def __init__(self, num_in=None, num_out=None, num_units: list = None, activate_func="ReLU", init=False,
                 expand_factor=4):
        super(AdaptiveFNN, self).__init__()
        if activate_func == "ReLU":
            base_layer = LinearReLU
        elif activate_func == "Sigmoid":
            base_layer = LinearSigmoid
        elif activate_func is None:
            base_layer = Linear
        else:
            raise RuntimeError("can't use this activate function")
        idx = 0
        if num_units is None:
            num_unit_in = num_in
            num_unit_out = int(num_unit_in / expand_factor)
            while num_unit_out > num_out:
                setattr(self, "linear{}".format(idx), base_layer(num_in=num_unit_in, num_out=num_unit_out, init=init))
                num_unit_in = num_unit_out
                num_unit_out = int(num_unit_in / expand_factor)
                idx += 1
            setattr(self, "linear{}".format(idx), base_layer(num_in=num_unit_in, num_out=num_out, init=init))
            self.last_layer = nn.Linear(num_out, num_out)
            self.num_layers = idx + 1
        else:
            for idx in range(0, len(num_units) - 1):
                setattr(self, "linear{}".format(idx),
                        base_layer(num_in=num_units[idx], num_out=num_units[idx + 1], init=init)
                        )
                idx += 1
            self.last_layer = nn.Linear(num_units[idx], num_units[idx])
            self.num_layers = idx

    def forward(self, x):
        out = x
        for idx in range(self.num_layers):
            out = getattr(self, "linear{}".format(idx))(out)
        return self.last_layer(out)
