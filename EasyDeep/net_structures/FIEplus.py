#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/3/4 16:04
# @Author  : strawsyz
# @File    : FIEplus.py
# @desc:

from torch import nn

from net_structures.old_models import FIE2
from nets.TorchvisionNet import create_model


class FIEISlowFastNet(nn.Module):
    def __init__(self, vocab_size=2048, embedding_dim=256, N=1, heads=1, max_seq_len=200, num_class=101):
        super().__init__()
        self.fie = FIE2(vocab_size, embedding_dim, N, heads, num_class, max_seq_len=200)
        self.backbone = create_model("slowfast_r50")
        self.softmax = nn.Softmax()

    def forward(self, data, features):
        output_fie = self.fie(features)
        output = self.backbone(data.permute(3, 0, 4, 1, 2))
        # output = self.softmax(output)
        # output = output.squeeze(dim=2)
        return output + output_fie
