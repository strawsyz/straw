# -*- coding: utf-8 -*-
# File  : 295-3.py
# Author: strawsyz
# Date  : 2023/6/5

# 使用SortedList来保存数组
# 参考公开的解法，有效但是效果一般

from sortedcontainers import SortedList

class MedianFinder:

    def __init__(self):
        self.nums = SortedList([])

    def addNum(self, num: int) -> None:
        self.nums.add(num)

    def findMedian(self) -> float:
        n = len(self.nums)
        if n%2==0:
            return (self.nums[n//2] +  self.nums[n//2-1])/2
        else:
            return self.nums[n//2]