import numpy as np


class Solution:
    def minMoves2(self, nums: List[int]) -> int:
        nums.sort()
        length = len(nums)
        median = nums[length//2]
        left, right = 0, length - 1
        # for i in nums:

        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] < median:
                left = mid + 1
            elif nums[mid] > median:
                right = mid - 1
            else:
                break
        if left > right:
            if abs(nums[left] - median) > abs(nums[right] - median):
                median = nums[right]
            else:
                median = nums[left]
        else:
            median = nums[mid]
        sum = 0
        for num in nums:
            sum += abs(num - median)
        return sum
