#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/3/2 13:11
# @Author  : strawsyz
# @File    : I3D.py
# @desc:
import torch

from base.base_net import BaseNet
from configs.net_config import VideoFeatureNetConfig
from net_structures.FIEI3D import FIEI3DNet, FIEI3DNet2, get_i3d_model
from pytorch_i3d import InceptionI3d


class PretrainedI3DNet(VideoFeatureNetConfig, BaseNet):
    def __init__(self):
        super(PretrainedI3DNet, self).__init__()
        self.net_structure = get_i3d_model()


class MyPretrainedI3DNet(VideoFeatureNetConfig, BaseNet):
    def __init__(self):
        super(MyPretrainedI3DNet, self).__init__()
        self.net_structure = FIEI3DNet(vocab_size=self.vocab_size, embedding_dim=self.embeddings_dim, N=self.N,
                                             heads=self.heads, pretrained=True)


class MyPretrainedI3DNet2(VideoFeatureNetConfig, BaseNet):
    def __init__(self):
        super(MyPretrainedI3DNet2, self).__init__()
        self.net_structure = FIEI3DNet2(vocab_size=self.vocab_size, embedding_dim=self.embeddings_dim, N=self.N,
                                             heads=self.heads, pretrained=True)



if __name__ == '__main__':
    data = torch.ones((15, 3, 13, 224, 224))
    model = get_i3d_model()
    out = model(data)
    print(out.shape)
