#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020/10/4 15:09
# @Author  : strawsyz
# @File    : LoanNet.py
# @desc:
from base.base_net import BaseNet
from net_structures.FNN import AdaptiveFNN
from configs.net_config import LoanNetConfig


class LoanNet(LoanNetConfig, BaseNet):
    def __init__(self):
        super(LoanNet, self).__init__()
        self.net_structure = AdaptiveFNN(num_in=self.n_in, num_out=self.n_out,
                                         num_units=[4, 16, 64, 256, 64, 32, 16, 8, 2], activate_func=None,
                                         init=True)
