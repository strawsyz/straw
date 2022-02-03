# 动态规划法
# 进一步优化空间复杂度
# 使用一维数组来存储
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        dp = [1] * m
        for _ in range(1, n):
            for col in range(1, m):
                dp[col] = dp[col - 1] + dp[col]
        return dp[-1]
