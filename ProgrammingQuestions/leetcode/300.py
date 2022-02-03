from typing import List


# 时间复杂度n^2
# 空间复杂的为n

# LIS问题
# 最长上升子序列

class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        if len(nums) == 0:
            return 0
        dp = [1] * len(nums)
        for i in range(1, len(nums)):
            # 两个循环
            dp[i] = max([1 + dp[j] for j in range(i) if nums[j] < nums[i]] + [1])
        return max(dp + [1])


if __name__ == '__main__':
    s = Solution()
    input = [10, 9, 2, 5, 3, 7, 101, 18]
    res = s.lengthOfLIS(input)
    print(res)
