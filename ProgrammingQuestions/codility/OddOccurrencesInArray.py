#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2021/12/23 11:10
# @Author  : strawsyz
# @File    : OddOccurrencesInArray START.py
# @desc:


def solution(A):
    if len(A) == 1:
        return A[0]
    a = 0
    for i in A:
        a ^= i
    return a


if __name__ == '__main__':
    print(6 ^ 6 ^ 5 ^ 5 ^ 1342342 ^ 0)
