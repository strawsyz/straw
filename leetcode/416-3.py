from typing import List


# 优化二
# 进一步优化，但是空间，时间复杂度没有太大变化

class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        all_sum = sum(nums)
        if all_sum % 2 != 0:
            return False
        target = all_sum // 2
        dp = [0] * (target + 1)
        dp[0] = 1

        for num in nums:
            # 采用逆序
            for j in range(target, num - 1, -1):
                dp[j] = dp[j] or dp[j - num]
        return dp[-1]


if __name__ == '__main__':
    s = Solution()
    input = [1, 5, 11, 5]
    res = s.canPartition(input)
    print(res)
