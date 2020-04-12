from typing import List
# 使用动态规划

class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        n_rows = len(grid)
        n_cols = len(grid[0])
        matrix = [[None for _ in range(n_cols)] for _ in range(n_rows)]
        for row in range(n_rows):
            for col in range(n_cols):
                if row == 0 and col == 0:
                    matrix[row][col] = grid[row][col]
                elif row == 0:
                    matrix[row][col] = matrix[row][col - 1] + grid[row][col]
                elif col == 0:
                    matrix[row][col] = matrix[row - 1][col] + grid[row][col]
                else:
                    matrix[row][col] = min(matrix[row][col - 1], matrix[row - 1][col]) + grid[row][col]
        return matrix[n_rows - 1][n_cols - 1]

