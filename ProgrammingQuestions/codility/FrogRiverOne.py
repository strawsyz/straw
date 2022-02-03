#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2021/12/23 12:12
# @Author  : strawsyz
# @File    : FrogRiverOne START.py
# @desc:

def solution(X, A):
    # write your code in Python 3.6
    if X > len(set(A)):
        return -1

    if len(A) == 1:
        if A[0] == 1:
            return 0
        else:
            return -1

    s = set()
    for idx, a in enumerate(A):
        s.add(a)
        if len(s) == X:
            return idx