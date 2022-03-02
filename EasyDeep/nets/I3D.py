#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/3/2 13:11
# @Author  : strawsyz
# @File    : I3D.py
# @desc:
import torch

from Args.video_args import get_args
from base.base_net import BaseNet
from configs.net_config import VideoFeatureNetConfig
from pytorch_i3d import InceptionI3d


class PretrainedI3DNet(VideoFeatureNetConfig, BaseNet):
    def __init__(self):
        super(PretrainedI3DNet, self).__init__()
        if get_args().system == "Linux":
            self.net_structure = get_i3d_model(r"/workspace/pytorch-i3d/models/rbg_imagenet.pt")
        else:
            self.net_structure = get_i3d_model()


def get_i3d_model(model_path=r"C:\(lab\OtherProjects\pytorch-i3d-master\models\rgb_imagenet.pt"):
    import platform
    if platform.system() == "Linux":
        model_path = r"/workspace/datasets/rgb_imagenet.pt"

    i3d = InceptionI3d(400, in_channels=3)
    # i3d.replace_logits(157)
    i3d.load_state_dict(torch.load(model_path))
    # i3d.cuda()
    return i3d

