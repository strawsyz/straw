#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020/10/15 22:18
# @Author  : strawsyz
# @File    : singleton4.py
# @desc:

class Singleton:
    def __init__(self):
        self.name = "singleton"


instance = Singleton()
del Singleton

print(instance.name)
# will get a error
instance = Singleton()