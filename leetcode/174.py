from typing import List


# 动态规划法，这题是自底向上
class Solution:
    def calculateMinimumHP(self, dungeon: List[List[int]]) -> int:
        n_row = len(dungeon)
        n_col = len(dungeon[0])

        dp = [[0 for _ in range(n_col)] for _ in range(n_row)]
        if dungeon[n_row - 1][n_col - 1] >= 0:
            dp[n_row - 1][n_col - 1] = 1
        else:
            dp[n_row - 1][n_col - 1] = 1 - dungeon[n_row - 1][n_col - 1]

        for row in range(n_row - 1, -1, -1):
            for col in range(n_col - 1, -1, -1):
                if row == n_row - 1 and col == n_col - 1:
                    continue
                if row == n_row - 1:
                    dp[row][col] = dp[row][col + 1] - dungeon[row][col]

                elif col == n_col - 1:
                    dp[row][col] = dp[row + 1][col] - dungeon[row][col]

                else:
                    # 选择扣血最少的路走
                    dp[row][col] = min(dp[row + 1][col] - dungeon[row][col], dp[row][col + 1] - dungeon[row][col])

                if dp[row][col] < 1:
                    dp[row][col] = 1
        return dp[0][0]


if __name__ == '__main__':
    s = Solution()
    res = s.calculateMinimumHP([[1, -4, 5, -99], [2, -2, -2, -1]])
    print(res)
