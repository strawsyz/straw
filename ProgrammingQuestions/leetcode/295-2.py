# 使用两个堆排序

# 参考官方的解法
class MedianFinder(object):

    def __init__(self):
        # 中位数，数组的长度单数还是偶数 两个顶堆
        self.med, self.odd, self.heaps = 0, 0, [[], []]

    def addNum(self, x):
        big, small = self.heaps
        if self.odd:
            # todo 这边还有待理解
            # 如果本来有单数个数字
            heapq.heappush(big, max(x, self.med))
            heapq.heappush(small, -min(x, self.med))
            self.med = (big[0] - small[0]) / 2.0
        else:
            # 如果本来有偶数个数字
            if x > self.med:
                # 如果大于原有的中位数，就插到大顶堆
                self.med = heapq.heappushpop(big, x)
            else:
                # 如果小于原有的中位数，就插到小顶堆
                self.med = -heapq.heappushpop(small, -x)
        # 改成单数或是偶数
        self.odd ^= True

    def findMedian(self):
        return self.med