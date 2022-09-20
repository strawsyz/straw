#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/3/4 16:04
# @Author  : strawsyz
# @File    : FIEplus.py
# @desc:

from torch import nn

from net_structures.old_models import FIE2
import torch

class FIEISlowFastNet(nn.Module):
    def __init__(self, vocab_size=2048, embedding_dim=256, N=1, heads=1, max_seq_len=200, num_class=101,
                 pretrained=True):
        super().__init__()
        self.fie = FIE2(vocab_size, embedding_dim, N, heads, num_class, max_seq_len=200)
        self.backbone = create_model("slowfast_r50", num_class=num_class, pretrained=pretrained)
        self.softmax = nn.Softmax()

    def forward(self, data, features):
        output_fie = self.fie(features)
        output = self.backbone(data)
        return output + output_fie


def create_model(model_name, num_class=101, pretrained=True):
    model = torch.hub.load("facebookresearch/pytorchvideo", model=model_name, pretrained=pretrained)

    layers = list(model.blocks.children())
    _layers = layers[:-1]
    # feature_extractor = nn.Sequential(*_layers)

    # 2. Classifier:
    fc = layers[-1]
    in_features = layers[-1].proj.in_features
    fc.proj = nn.Linear(in_features=in_features, out_features=num_class, bias=True)
    return model
