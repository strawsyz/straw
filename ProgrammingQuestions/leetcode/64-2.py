# 使用动态规划 + 滚动数组 空间消耗更加少
# 滚动数组存储当前行和上一行的内容

class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        n_rows = len(grid)
        n_cols = len(grid[0])
        matrix = [[None for _ in range(n_cols)] for _ in range(2)]
        now = 0

        for row in range(n_rows):
            old = now
            now = 1 - now
            for col in range(n_cols):
                if row == 0 and col == 0:
                    matrix[now][col] = grid[row][col]
                elif row == 0:
                    matrix[now][col] = matrix[now][col - 1] + grid[row][col]
                elif col == 0:
                    matrix[now][col] = matrix[old][col] + grid[row][col]
                else:
                    matrix[now][col] = min(matrix[now][col - 1], matrix[old][col]) + grid[row][col]

        return matrix[now][n_cols - 1]
