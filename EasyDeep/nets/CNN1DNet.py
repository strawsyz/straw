#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2020/7/21 14:41
# @Author  : Shi
# @FileName: CNN1DNet.py
# @Description：

from base.base_net import BaseNet
from net_structures.CNN1D import MyCNN1D
from configs.net_config import CNN1DNetConfig

class CNN1DNet(BaseNet,CNN1DNetConfig):
    def __init__(self, config_instance=None):
        super(CNN1DNet, self).__init__(config_instance)
        # init network
        self.net_structure = MyCNN1D()


if __name__ == '__main__':
    base_net = CNN1DNet()
    base_net.get_net(base_net, False)
