#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2020/7/15 13:51
# @Author  : Shi
# @FileName: FNN.py
# @Description：
from torch import nn


class LinearReLU(nn.Sequential):
    def __init__(self, num_in, num_out):
        super(LinearReLU, self).__init__()
        self.add_module("Linear", nn.Linear(num_in, num_out))
        self.add_module("ReLU", nn.ReLU(inplace=True))


class LinearSigmoid(nn.Sequential):
    def __init__(self, num_in, num_out):
        super(LinearSigmoid, self).__init__()
        self.add_module("Linear", nn.Linear(num_in, num_out))
        self.add_module("Sigmoid", nn.Sigmoid())


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


class MyFNN(nn.Module):
    def __init__(self, n_in, n_out, dropout=0.05):
        super(MyFNN, self).__init__()
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
    def __init__(self, num_in=None, num_out=None, num_units: list = None, activate_func="ReLU"):
        super(AdaptiveFNN, self).__init__()
        if activate_func == "ReLU":
            base_layer = LinearReLU
        elif activate_func == "sigmoid":
            base_layer = LinearSigmoid
        idx = 0
        if num_units is None:
            expand = 4
            num_unit_in = num_in
            num_unit_out = int(num_unit_in / expand)
            while num_unit_out > num_out:
                setattr(self, "linear{}".format(idx), base_layer(num_in=num_unit_in, num_out=num_unit_out))
                num_unit_in = num_unit_out
                num_unit_out = int(num_unit_in / expand)
                idx += 1
            setattr(self, "linear{}".format(idx), base_layer(num_in=num_unit_in, num_out=num_out))
            self.last_layer = nn.Linear(num_out, num_out)
            self.num_layers = idx + 1
        else:
            for idx in range(0, len(num_units) - 1):
                setattr(self, "linear{}".format(idx), base_layer(num_in=num_units[idx], num_out=num_units[idx + 1]))
                idx += 1
            self.last_layer = nn.Linear(num_units[idx], num_units[idx])
            self.num_layers = idx

    def forward(self, x):
        out = x
        for idx in range(self.num_layers):
            out = getattr(self, "linear{}".format(idx))(out)
        return self.last_layer(out)


if __name__ == '__main__':
    import torch

    data = torch.randn((16, 8))
    model = AdaptiveFNN(num_units=[8, 8, 2])
    out = model(data)
    print(out)
    print(out.shape)
