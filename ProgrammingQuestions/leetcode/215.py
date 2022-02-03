from typing import List
import heapq

# 使用Python自带的排序
class Solution():
    def findKthLargest(self, nums: List[int], k: int) -> int:
        return sorted(nums, reverse=True)[k - 1]

