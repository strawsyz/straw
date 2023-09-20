# -*- coding: utf-8 -*-
# File  : DD10 xor.py
# Author: strawsyz
# Date  : 2023/9/20

# 描述
# 给出n个数字 a_1,...,a_n，问最多有多少不重叠的非空区间，使得每个区间内数字的xor都等于0。
# 输入描述：
# 第一行一个整数ｎ； 第二行ｎ个整数　a_1,...,a_n； 对于30%的数据，n<=20； 对于100%的数据，n<=100000, a_i<=100000；
# 输出描述：
# 一个整数表示最多的区间个数；

import sys


inputs = []
for line in sys.stdin:
    a = line.split()
    inputs.append(a)

n = inputs[0]
nums = inputs[1]
nums = [int(num) for num in nums]

res = 0
dp = [0]
for num in nums:
    if num == 0:
        res += 1
        dp = [0]
        continue
    else:
        dp = [i^num for i in dp]
        if 0 in dp:
            res +=1
            dp = [0]
        else:
            dp.append(num)
print(res)
