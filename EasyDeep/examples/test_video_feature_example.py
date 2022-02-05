#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020/9/22 8:57
# @Author  : strawsyz
# @File    : test_mnist_example.py
# @desc:
from unittest import TestCase

from experiments.video_feature_experiment import VideoFeatureExperiment


class TestMnistExperiment(TestCase):
    def test_valid_one_batch(self):
        # config = VideoFeatureConfig()
        experiment = VideoFeatureExperiment()
        print(experiment.recorder)
        print(experiment.experiment_record)
        # experiment.test()
        # experiment.estimate()
        # experiment.save_history()
        # print("end")
