import heapq


# 能获得中位数，但是不能应对有负数存在的情况
# -*- coding:utf-8 -*-
class Solution:
    # 记录当前数字的是否是奇数，1表示是奇数，0不表示不是
    is_single = 0
    # 保存一个最大堆，全都是负数
    max_heap = []
    # 保存一个最小堆，都是正数
    min_heap = []

    def Insert(self, num):
        self.is_single = 1 - self.is_single
        if self.is_single:
            # 如果当前有奇数个数字
            # 将当前的数字添加到最小堆里面
            heapq.heappush(self.min_heap, num)
            # 将最小堆的根节点的负数扔到最大堆里面
            heapq.heappush(self.max_heap, -heapq.heappop(self.min_heap))
        else:
            # 如果当前有偶数个数字
            # 将当前的数字的负数放入最大堆里面
            heapq.heappush(self.max_heap, -num)
            # 将最大堆的根节点的负数放到最小堆里面
            heapq.heappush(self.min_heap, -heapq.heappop(self.max_heap))

    # 本函数如果不多添加一个参数就会报错，网站的一个bug
    def GetMedian(self, _):
        # write code here
        if self.is_single:
            return -self.max_heap[0]
        else:
            return (-self.max_heap[0] + self.min_heap[0]) / 2.0

