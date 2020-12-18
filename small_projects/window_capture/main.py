#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020/12/19 0:19
# @Author  : strawsyz
# @File    : main.py
# @desc:


from win_utils import *
"""
capture window's shot
"""
if __name__ == '__main__':
    show_all_windows()
    capture_window(class_name="SunAwtFrame", save_path="window_shot.png")
