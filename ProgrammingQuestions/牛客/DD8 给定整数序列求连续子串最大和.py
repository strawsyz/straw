# -*- coding: utf-8 -*-
# File  : DD8 给定整数序列求连续子串最大和.py
# Author: strawsyz
# Date  : 2023/9/14


import sys

nums = []
for line in sys.stdin:
    nums = line.split()
nums = [int(i) for i in nums]
res = nums[0]  # -sys.maxsize-1
sum = 0
for num in nums:
    sum += num
    if sum > 0:
        res = max(sum,res)
    else:
        sum = 0
print(res)


