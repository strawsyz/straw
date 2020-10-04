#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2020/7/21 14:41
# @Author  : Shi
# @FileName: CNN1DNet.py
# @Description：

from base.base_net import BaseNet
from net_structures.CNN1D import MyCNN1D
from configs.net_config import CNN1DBaseNetConfig


class CNN1DNet(BaseNet, CNN1DBaseNetConfig):
    def __init__(self):
        super(CNN1DNet, self).__init__()
        # init network
        self.net_structure = MyCNN1D()


if __name__ == '__main__':
    base_net = CNN1DNet()
    base_net.get_net(base_net, False)
