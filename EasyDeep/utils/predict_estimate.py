#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/10/8 20:47
# @Author  : strawsyz
# @File    : test_image_example.py
# @desc:
from unittest import TestCase

from examples.image_example import ImageExperiment
from utils.estimate_utils import iou_estimate


def pretrained_model_predict_test_dataset(experiment, pretrain_path):
    experiment.pretrain_path = pretrain_path.strip()
    experiment.is_pretrain = True
    result_path = experiment.test(save_predict_result=True)
    return result_path


if __name__ == '__main__':
    gt_data_path = r""
    pretrain_path = r""

    experiment = ImageExperiment()
    image_path = r""
    mask_path = r""
    experiment.pretrain_path = pretrain_path.strip()
    experiment.is_pretrain = True

    result_path = pretrained_model_predict_test_dataset(experiment, pretrain_path)
    iou_estimate(gt_data_path, result_path)
