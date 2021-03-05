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
    # trained model's path
    pretrain_path = r"/home/shi/Downloads/models/polyp/2021-03-04/ep21_20-05-02.pkl"
    experiment = ImageExperiment()
    pretrained_model_predict_test_dataset(experiment, pretrain_path)

    # experiment.history_save_path = r""
    # experiment.estimate(save_path="tmp.png", use_log10=True)

    # predict_data_path = r'/home/shi/Download\models\polyp/result/trained_resnet34'
    # predict_data_path = r'/home/shi/Download\models\polyp/result/resnet50-unpretrained-01'

    gt_data_path = r"/home/shi/Downloads/dataset/polyp/TMP/09/test/mask"
    result_path = r"/home/shi/Download\models\polyp/result/2021-03-04/"
    iou_estimate(gt_data_path, result_path)