#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2021/12/26 14:19
# @Author  : strawsyz
# @File    : MaxNonoverlappingSegments.py
# @desc:


def solution(A, B):
    if len(A) == 0:
        return 0
    if len(A) == 1:
        return 1
    end = B[0]
    count = 1
    for i in range(1, len(B)):
        if A[i] > end:
            count += 1
            end = B[i]

    return count
