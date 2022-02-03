# 也使用滑动窗口。稍微有点变化

class Solution:
    def minSubArrayLen(self, s: int, nums: List[int]) -> int:
        left, right, min_length = 0, 0, len(nums) + 1
        sums = []
        for num in nums:
            if not sums:
                sums.append(num)
            else:
                sums.append(sums[-1] + num)
        while left < len(nums) and right < len(nums):
            if sums[right] - sums[left] + nums[left] < s:
                right += 1
            else:
                if right + 1 - left < min_length:
                    min_length = right + 1 - left
                left += 1
        if min_length != len(nums) + 1:
            return min_length
        else:
            return 0
