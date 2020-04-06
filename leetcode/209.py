# 给定一个含有 n 个正整数的数组和一个正整数 s ，找出该数组中满足其和 ≥ s 的长度最小的连续子数组。
# 如果不存在符合条件的连续子数组，返回 0。
from typing import *


# 使用滑动窗口
class Solution:
    def minSubArrayLen(self, s: int, nums: List[int]) -> int:
        left = right = 0
        length = len(nums) - 1
        min_len = 0
        sum_temp = nums[0]

        while True:
            if sum_temp < s:
                if right < length:
                    right += 1
                    sum_temp = sum_temp + nums[right]
                else:
                    break
            if sum_temp >= s:
                if min_len > right - left + 1 or min_len == 0:
                    min_len = right - left + 1
                sum_temp -= nums[left]
                left += 1
        return min_len


if __name__ == '__main__':
    s = Solution()
    res = s.minSubArrayLen(7, [2, 3, 1, 2, 4, 3])
    print(res)
