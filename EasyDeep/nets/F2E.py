#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/2/7 12:01
# @Author  : strawsyz
# @File    : F2E.py
# @desc:
from base.base_net import BaseNet
from configs.net_config import VideoFeatureNetConfig
from net_structures.old_models import FIE2


class FIE2Net(VideoFeatureNetConfig, BaseNet):
    def __init__(self):
        super(FIE2Net, self).__init__()
        self.net_structure = FIE2(self.n_in, self.embeddings_dim, self.n_layer, self.heads, self.n_out)
