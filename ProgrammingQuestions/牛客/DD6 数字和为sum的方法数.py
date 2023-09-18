# -*- coding: utf-8 -*-
# File  : DD6 数字和为sum的方法数.py
# Author: strawsyz
# Date  : 2023/9/14


import sys

inputs = []
for line in sys.stdin:
    a = line.split()
    inputs.append(a)

n, sum = int(inputs[0][0]), int(inputs[0][1])
A = [int(i) for i in inputs[1]]
dp = [0 for i in range(sum+1)]  # 达到各个target值的方法数
dp[0]=1  # 达到0的方法数默认为1
for i in range(n):
    # 每一个循环计算，加上下一个数字之后的计算结果
    sum_t = sum
    while sum_t>=A[i]: # 如果单个数字小于等于目标值
        # 将差值的方法数加上原本的方法数
        dp[sum_t] += dp[sum_t-A[i]]
        # 减少目标值
        sum_t -=1
print(dp[sum])