#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2021/06/21 3:04
# @Author  : strawsyz
# @File    : line2021.py
# @desc:


import  numpy as np

import math

te = -0.3+0.53+0.46-0.77+0.9-0.4
print(te)
te = 0.46-0.39-0.75+0.3-0.56+0.55
print(te)

te = -0.77+0.51-0.44-0.42+0.2+0.51
print(te)
te = -0.97+0.49-0.86-0.17+0.22+0.13
print(te)

te = np.sqrt((0.6+0.41)**2 + (1.46)**2)
print(te)


# if first <x :
#     first = x
#     second = first
# elif second < x:
#     second = x

te = 2.5**2 + 1.5**2 + 0.5**2
print(te/3)

def main(x):
    print((1+2*x+3*x**2+4*x**3+5*x**4+6*x**5+7*x**6+8*x**7+9*x**8+10*x**9)*(1-x))

main(0.2)

# print(2.4244172799999997/1.25)