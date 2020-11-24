#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/10/8 20:47
# @Author  : strawsyz
# @File    : test_image_example.py
# @desc:
from unittest import TestCase

from examples.image_example import ImageExperiment

if __name__ == '__main__':
    experiment = ImageExperiment()
    # trained vgg16
    pretrain_path = ""

    experiment.pretrain_path = pretrain_path.strip()
    experiment.is_pretrain = True
    experiment.test()
    experiment.history_save_path = r""
    experiment.estimate(save_path="tmp.png", use_log10=True)
