#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/2/10 17:08
# @Author  : strawsyz
# @File    : other_utils.py
# @desc:
import os


def set_GPU(gpu_ids: str):
    if gpu_ids != "-1":
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
