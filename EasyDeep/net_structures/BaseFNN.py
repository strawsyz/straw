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
    def __init__(self, n_in, n_out, dropout=0.05):
        super(MyFNN, self).__init__()
        self.linears = []
        for i in range(8):
            from operator import attrgetter
            attr = attrgetter("linear{}".format(i))
            setattr(self, "linear{}".format(i),
                    DenseLayer(n_in=n_in, n_out=n_out, n_hides=[2048, 1024, 512, 256, 128, 64, 32, 16],
                               dropout=dropout))
            self.linears.append(attr(self))

    def forward(self, input):
        input = input.reshape(8, -1)
        output = self.linears[0](input[0])
        for i in range(1, 8):
            output += self.linears[i](input[i])
        return output
# [INFO]<2020-07-29 14:20:58,394> ecg_sample.py->train_one_epoch line:45 { EPOCH:346	 train_loss:0.040702	 }
# [INFO]<2020-07-29 14:20:59,599> ecg_sample.py->train_one_epoch line:59 { Epoch346:	 valid_loss:0.588721 }
# [INFO]<2020-07-29 14:20:59,600> deep_experiment.py->save_history line:126 { ========== saving history========== }
# [INFO]<2020-07-29 14:20:59,622> deep_experiment.py->save_history line:135 { ========== saved history at D:\models\Ecg\history_1595399946.pth========== }
# [INFO]<2020-07-29 14:20:59,622> deep_experiment.py->save line:98 { save this model for ['train_loss', 'valid_loss'] is better }
# [INFO]<2020-07-29 14:20:59,622> deep_experiment.py->_save_model line:104 { ==============saving model data=============== }
# [INFO]<2020-07-29 14:21:04,992> deep_experiment.py->_save_model line:108 { ==============saved at D:\models\Ecg\2020-07-29\ep346_14-20-59.pkl=============== }