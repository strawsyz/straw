#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020/10/4 15:09
# @Author  : strawsyz
# @File    : MultiLabelNet.py
# @desc:
from base.base_net import BaseNet
from configs.net_config import MultiLabelNetConfig
from net_structures.CNN1D import MultiLabelCNN1D
from net_structures.FNN import AdaptiveFNN


class MultiLabelNet(MultiLabelNetConfig, BaseNet):
    def __init__(self):
        super(MultiLabelNet, self).__init__()
        self.net_structure = MultiLabelCNN1D()
