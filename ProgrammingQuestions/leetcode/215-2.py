from typing import List
import heapq

# 使用堆排序,选出第k大的数字
class Solution():
    def findKthLargest(self, nums: List[int], k: int) -> int:
        return heapq.nlargest(k, nums)[-1]
