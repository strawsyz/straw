# 思路：使用一个大顶堆和一个小顶堆来保存数组

# 参考官方的解法
class MedianFinder(object):

    def __init__(self):
        # 中位数，数组的长度单数还是偶数 两个顶堆
        self.med, self.odd, self.heaps = 0, 0, [[], []]

    def addNum(self, x):
        big, small = self.heaps
        if self.odd:  # 来本是奇数状态，添加数字之后变成偶数
            # 将两个堆的最大值和最小值提取出来，进行平均
            heapq.heappush(big, max(x, self.med))
            heapq.heappush(small, -min(x, self.med))
            self.med = (big[0] - small[0]) / 2.0
        else:
            if x > self.med:
                # 如果大于原有的中位数，就插到大顶堆
                self.med = heapq.heappushpop(big, x)
            else:
                # 如果小于原有的中位数，就插到小顶堆
                self.med = -heapq.heappushpop(small, -x)
        # 修改奇偶数状态
        self.odd ^= True

    def findMedian(self):
        return self.med