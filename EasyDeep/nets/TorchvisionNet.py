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
from net_structures.FIEplus import FIEISlowFastNet


class SlowFastNet(VideoFeatureNetConfig, BaseNet):
    def __init__(self):
        super(SlowFastNet, self).__init__()
        self.net_structure = create_model("slowfast_r50")


class FIESlowFast(VideoFeatureNetConfig, BaseNet):
    def __init__(self):
        super(FIESlowFast, self).__init__()
        self.net_structure = FIEISlowFastNet(vocab_size=self.vocab_size,embedding_dim=self.embedding_dim,N=self.N, heads=self.heads, num_class=101)


def create_model(model_name, num_target_classes=101, pretrained=True):
    model = torch.hub.load("facebookresearch/pytorchvideo", model=model_name, pretrained=pretrained)

    layers = list(model.blocks.children())
    _layers = layers[:-1]
    # feature_extractor = nn.Sequential(*_layers)

    # 2. Classifier:
    fc = layers[-1]
    in_features = layers[-1].proj.in_features
    fc.proj = nn.Linear(in_features=in_features, out_features=num_target_classes, bias=True)
    return model
