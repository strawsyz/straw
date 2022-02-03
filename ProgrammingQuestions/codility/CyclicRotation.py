#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2021/12/23 10:40
# @Author  : strawsyz
# @File    : CyclicRotation.py
# @desc:


def solution(A, K):
    length = len(A)
    if length == K or K == 0 or length == 0:
        return A
    K = K % length

    B = A[-K:] + A[:-K]
    return B


if __name__ == '__main__':
    print(solution([3, 8, 9, 7, 6], 6))
