#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020/9/21 23:16
# @Author  : strawsyz
# @File    : base_config.py
# @desc: load system information and global variables
import platform
import string
import random
from base.base_logger import BaseLogger
import torch

from mixins.system_config_mixin import SystemConfigMixin
from abc import ABC, abstractmethod
from utils.time_utils import get_date
from mixins.mysql_mixin import MySQLMixin


class BaseConfig(BaseLogger, SystemConfigMixin, MySQLMixin):
    def __init__(self):
        super(BaseConfig, self).__init__()
        MySQLMixin.__init__(self)
        SystemConfigMixin.__init__(self)
        self.hide_attrs_4_gui = ["config_info"]
        # self._system = platform.system()

    def get_attrs_4_gui(self):
        attrs = []
        hidden_attrs = self.system_config + self.hide_attrs_4_gui
        for attr in self.__dict__:
            if attr not in hidden_attrs:
                attrs.append((attr, self.__dict__.get(attr)))
        return attrs

    def get_attrs_4_print(self):
        attrs = []
        hidden_attrs = self.system_config + self.hide_attrs_4_gui
        for attr in self.__dict__:
            if attr not in hidden_attrs:
                attrs.append(attr)
        print(attrs)
        return attrs
        # return self.get_attrs_4_gui()


class BaseDataSetConfig(BaseConfig):
    def __init__(self):
        super(BaseDataSetConfig, self).__init__()
        self.random_state = None
        self.test_model = False

    def __str__(self):
        return __class__.__name__


class BaseExperimentConfig(BaseConfig, ABC):
    def __init__(self, tag=None):
        super(BaseExperimentConfig, self).__init__()
        self.use_prettytable = False
        self._system = platform.system()
        self.dataset_config = None
        self.net_config = None
        self.tag = tag
        self.history_save_path = None
        # is use database to save
        self.use_db = False

    @abstractmethod
    def set_dataset(self):
        pass

    @abstractmethod
    def set_net(self):
        pass

    @abstractmethod
    def set_model_selector(self):
        pass

    def create_random_str(self, length=6):
        return ''.join(random.sample(string.ascii_letters + string.digits, length))

    def create_experiment_name(self, name: str, tag: str = None):
        experiment_name = name
        if tag is not None:
            experiment_name += tag
        experiment_name = "{}-{}-{}".format(experiment_name, get_date(format_="%m-%d"), self.create_random_str())
        self.experiment_name = experiment_name
        return experiment_name


class BaseNetConfig(BaseConfig):
    def __init__(self):
        super(BaseNetConfig, self).__init__()
        # net structure config
        self.n_out = 1
        # other net config
        self.loss_func_name = "BCEWithLogitsLoss"
        self.optim_name = "adam"
        self.is_scheduler = True
        self.scheduler_step_size = 15
        self.scheduler_gamma = 0.8
        self.lr = 0.002
        self.weight_decay = 0.001
        self.init_attr()

    def init_attr(self):
        # initialize some attributes
        if self.scheduler_gamma is None:
            self.scheduler_gamma = 0.1

    def __str__(self):
        return __class__.__name__


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
