# 简单的暴力解法
# 但是会超时

class MedianFinder:
    def __init__(self):
        """
        initialize your data structure here.
        """
        self.nums = []

    def addNum(self, num: int) -> None:
        index = bisect.bisect_left(self.nums, num)
        if index != len(self.nums):
            temp = self.nums[-1]
            for i in range(len(self.nums)-1,index,-1):
                self.nums[i] = self.nums[i-1]
            self.nums[index] = num
            self.nums.append(temp)
        else:
            self.nums.append(num)

    def findMedian(self) -> float:
        length = len(self.nums)
        if length%2:
            return self.nums[length//2]
        else:
            return (self.nums[length // 2] + self.nums[(length // 2)-1])/2

# Your MedianFinder object will be instantiated and called as such:
# obj = MedianFinder()
# obj.addNum(num)
# param_2 = obj.findMedian()
