#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2021/04/10 11:24
# @Author  : strawsyz
# @File    : print_color.py
# @desc:


class bcolors:
    # https://godoc.org/github.com/whitedevops/colors
    BGRED = '\033[41m'
    BGGREEN = '\033[42m'
    BGYELLOW = '\033[43m'
    BGBLUE = '\033[44m'
    BGMAGENDA = '\033[45m'
    BGCYAN = '\033[46m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENDA = '\033[95m'
    CYAN = '\033[96m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def red(message:str):
    return bcolors.RED + message + bcolors.ENDC


def green(message:str):
    return bcolors.GREEN + message + bcolors.ENDC


def print_red(message):
    print(bcolors.RED + str(message) + bcolors.ENDC)


def print_blue(message):
    print(bcolors.BLUE + str(message) + bcolors.ENDC)


def print_green(message):
    print(bcolors.GREEN + str(message) + bcolors.ENDC)


def print_list(list_):
    list_ = [str(i) for i in list_]
    print("\n".join(list_))
