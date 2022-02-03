# 动态规划法
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        dp = [[0 for _ in range(m)] for _ in range(n)]

        # m 是列数  n是行数
        for col in range(m):
            for row in range(n):
                if col == 0 or row == 0:
                    dp[row][col] = 1
                else:
                    dp[row][col] = dp[row - 1][col] + dp[row][col - 1]
        return dp[n - 1][m - 1]
