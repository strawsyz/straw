# 会超时
class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        max_row = len(obstacleGrid) - 1
        max_col = len(obstacleGrid[0]) - 1
        moves = [[1, 0], [0, 1]]

        if obstacleGrid[0][0] == 1:
            return 0

        def is_over(row, col, max_row, max_col):
            if row > max_row or col > max_col:
                return True
            else:
                return False

        def dfs(row, col, max_row, max_col):
            result = 0
            if row == max_row and col == max_col:
                return 1
            for move in moves:
                next_row, next_col = row + move[0], col + move[1]
                # m 是列数  n是行数
                if is_over(next_row, next_col, max_row, max_col):
                    continue
                if obstacleGrid[next_row][next_col] == 1:
                    continue
                result += dfs(next_row, next_col, max_row, max_col)
            return result

        return dfs(0, 0, max_row, max_col)
