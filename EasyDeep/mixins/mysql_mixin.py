#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020/9/24 20:30
# @Author  : strawsyz
# @File    : mysql_mixin.py
# @desc:
from utils.mysql_utils import get_db_utils


class MySQLMixin:
    def __init__(self):
        self.db_utils = get_db_utils()
