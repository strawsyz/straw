#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2021/12/23 15:12
# @Author  : strawsyz
# @File    : ASDAS.py
# @desc:
def get_blocks(A, middle):
    sum_ = 0
    n_block = 0
    for a in A:
        sum_ += a
        if sum_ > middle:
            n_block += 1
            sum_ = a
    return n_block + 1


def solution(K, M, A):
    # write your code in Python 3.6
    if len(A) == 1:
        return A[0]
    if K == 1:
        return sum(A)
    if K >= len(A):
        return max(A)
    else:
        max_num = max(A)
        sum_num = sum(A)
        while max_num <= sum_num:
            middle = int((sum_num + max_num) / 2)
            blocks_count = get_blocks(A, middle)
            if K >= blocks_count:
                sum_num = middle - 1
            else:
                max_num = middle + 1
        return max_num
