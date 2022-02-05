#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/2/3 18:22
# @Author  : strawsyz
# @File    : Transformer.py
# @desc:
from base.base_net import BaseNet
from configs.net_config import VideoFeatureNetConfig
from net_structures.Transformer import Transformer2E


class TranformerNet(VideoFeatureNetConfig, BaseNet):
    def __init__(self):
        super(TranformerNet, self).__init__()
        self.net_structure = Transformer2E(self.n_in, self.embeddings_dim, self.n_layer, self.heads, self.n_out)
