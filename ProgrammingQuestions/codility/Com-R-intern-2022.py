#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2021/12/26 15:10
# @Author  : strawsyz
# @File    : rakuten-intern.py
# @desc:


def solution(A, K):
    n = len(A)
    best = 0
    count = 0
    for i in range(n - K - 1):
        if (A[i] == A[i + 1]):
            count = count + 1
        else:
            count = 0
        best = max(best, count)
    result = best + 1 + K

    return min(result,n)


if __name__ == '__main__':
    A = [1, 1, 3,3,3,3,3,3,3,3]
    A = [1,  5,5]
    K = 0

    res = solution(A, K)
    print(res)
