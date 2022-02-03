# 更聪明的做法

class Solution:
    def minMoves2(self, nums: List[int]) -> int:
        nums.sort()
        left, right = 0, len(nums) - 1
        # for i in nums:
        res = 0
        while left < right:
            res += nums[right] - nums[left]
            left += 1
            right -= 1
        return res
