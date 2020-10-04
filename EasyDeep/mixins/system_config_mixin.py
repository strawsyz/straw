#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020/9/25 0:27
# @Author  : strawsyz
# @File    : system_config_mixin.py
# @desc:
import platform
import psutil
import torch


class SystemConfigMixin:
    def __init__(self):
        self.python_version = platform.python_version()
        self.architecture = platform.architecture()
        self.computer_name = platform.node()
        self.platform_name = platform.platform()
        self.processor = platform.processor()
        self.python_build = platform.python_build()
        self.python_compiler = platform.python_compiler()
        self.system = platform.system()
        self.system_version = platform.version()
        self.system_alias = platform.system_alias(platform.system(), platform.system(), platform.version())
        user_str = " user_infos: \n "
        user_status = psutil.users()
        for item in user_status:
            user_str += str(item) + "\n"
        self.user_info = user_str

        system_configs = [attr for attr in self.__dict__]
        system_configs.append("system_config")
        self.system_config = system_configs

        self.use_db = False


def system_info():
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
