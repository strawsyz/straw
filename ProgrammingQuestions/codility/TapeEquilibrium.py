#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2021/12/23 11:42
# @Author  : strawsyz
# @File    : TapeEquilibrium START.py
# @desc:

def solution(A):
    if len(A) == 2:
        return abs(A[0] - A[1])

    s = sum(A)
    a = 0
    min_d = 10 ** 10
    for i in range(len(A) - 1):
        a += A[i]
        b = s - a
        if min_d > abs(a - b):
            min_d = abs(a - b)
    return min_d