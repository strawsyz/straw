# -*- coding: utf-8 -*-
# File  : DD4 末尾0的个数 .py
# Author: strawsyz
# Date  : 2023/9/14


import sys

inputs = []
for line in sys.stdin:
    a = line.split()
    inputs.append(a)

n = int(inputs[0][0])
res = 0
num_2 = 0
num_5 = 0
flag_10=False
for i in range(1,n+1):

    while i %10==0 and i!=0:
        res += 1
        i = i//10
    while i%2==0:
        num_2 += 1
        i = i//2
    while i % 5 ==0:
        num_5 += 1
        i = i//5
res += min(num_2, num_5)
print(res)
