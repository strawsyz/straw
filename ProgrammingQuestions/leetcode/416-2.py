from typing import List


# 优化一
# 不使用矩阵而是使用一维数组来存储

class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        all_sum = sum(nums)
        if all_sum % 2 != 0:
            return False
        target = all_sum // 2
        dp = [0] * (target + 1)
        dp[0] = 1

        if nums[0] <= target:
            dp[nums[0]] = 1
        for i in range(1, len(nums)):
            num = nums[i]
            # 采用逆序
            for j in range(target, 0, -1):
                if num <= j:
                    dp[j] = dp[j] or dp[j - num]
        return dp[-1]


if __name__ == '__main__':
    s = Solution()
    input = [1, 5, 11, 5]
    res = s.canPartition(input)
    print(res)
