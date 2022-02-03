# 和第一版本一样的思路
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        n_rows = len(grid)
        n_cols = len(grid[0])

        dp = [[None for _ in range(n_cols)] for _ in range(n_rows)]
        dp[0][0] = grid[0][0]
        for i in range(1, n_rows):
            dp[i][0] = dp[i - 1][0] + grid[i][0]
        for j in range(1, n_cols):
            dp[0][j] = dp[0][j - 1] + grid[0][j]
        for row in range(1, n_rows):
            for col in range(1, n_cols):
                dp[row][col] = min(dp[row - 1][col], dp[row][col - 1]) + grid[row][col]
        return dp[n_rows - 1][n_cols - 1]