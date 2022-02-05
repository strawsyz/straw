#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020/10/15 17:44
# @Author  : strawsyz
# @File    : base_config_item.py
# @desc:
from enum import Enum


class ConfigItemType(Enum):
    SYSTEM = 0
    DATASET = 1
    NET = 2
    EXPERIMENT = 3
    OTHER = 4


class ConfigItemHiddenLevel(Enum):
    GUI = 0
    DEBUG = 1
    INFO = 2


class BaseConfigItem:
    def __init__(self, name, value, type, group=None, desc=None, hidden_level: str = ""):
        self.name = name
        self.value = value
        self.type = type
        self.group = group
        self.desc = desc
        self.hidden_level = hidden_level

    def __str__(self):
        return "[{}]:{}".format(self.name, self.value)

