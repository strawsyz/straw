#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2020/7/26 1:00
# @Author  : Shi
# @FileName: system_config.py
# @Description：

# 添加系统的设置信息
# 根据当前所在的ip，mac地址之类的判断当前所在的服务器
# 然后进行相应的初始化
from platform import system
import torch

system = system()


# todo 添加判断是哪台机器的逻辑，根据机器决定一些设定
def info():
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        num_gpu = torch.cuda.device_count()
        devices = {}
        for i in range(num_gpu):
            device_name = torch.cuda.get_device_name(i)
            devices[i] = device_name
        print("cpu num is {}".format(num_gpu))
        for device in devices:
            print(device)
            print(devices.get(device))
    else:
        print("no gpu is available")


def current_device():
    current_device = torch.cuda.current_device()
    print(current_device)
    return current_device


if __name__ == '__main__':
    num_gpu = torch.cuda.device_count()
    print(num_gpu)
    print(torch.version.cuda)