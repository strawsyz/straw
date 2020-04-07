# -*- coding:utf-8 -*-
# 使用数组保存之前的计算结果，减少重复计算

class Solution:
    def multiply(self, A):
        A_length = len(A)
        B_left, B_right = [1] * A_length, [1] * A_length
        for i in range(A_length - 1):
            B_left[i + 1] = B_left[i] * A[i]
        for i in range(A_length - 2, -1, -1):
            B_right[i] = B_right[i + 1] * A[i + 1]
        B = [1] * A_length
        for i in range(A_length):
            B[i] = B_left[i] * B_right[i]
        return B
