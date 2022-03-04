#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/3/4 15:51
# @Author  : strawsyz
# @File    : SlowFastnet.py
# @desc:
import torch
from torch import nn

from base.base_net import BaseNet
from configs.net_config import VideoFeatureNetConfig
from net_structures.FIEplus import FIEISlowFastNet, create_model


class SlowFastNet(VideoFeatureNetConfig, BaseNet):
    def __init__(self):
        super(SlowFastNet, self).__init__()
        self.net_structure = create_model("slowfast_r50", pretrained=True)


class FIESlowFast(VideoFeatureNetConfig, BaseNet):
    def __init__(self):
        super(FIESlowFast, self).__init__()
        self.net_structure = FIEISlowFastNet(vocab_size=self.vocab_size, embedding_dim=self.embedding_dim, N=self.N,
                                             heads=self.heads, num_class=101, pretrained=True)
