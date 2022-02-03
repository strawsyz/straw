#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2021/12/23 10:34
# @Author  : strawsyz
# @File    : bingap.py
# @desc:

def solution(N):
    # write your code in Python 3.6
    # pass
    find_one = False
    max_length_zero = 0
    length_zero = 0
    for i in range(N):
        num = 2 ** (i + 1)
        if num > N:
            break
        if N % num == 0:
            if find_one:
                length_zero += 1
                if length_zero > max_length_zero:
                    max_length_zero = length_zero

        else:
            find_one = True
            length_zero = 0
            N -= 2 ** i
    return max_length_zero


if __name__ == '__main__':
    print(solution(1041))
