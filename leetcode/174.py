from typing import List


# 动态规划法
class Solution:
    def calculateMinimumHP(self, dungeon: List[List[int]]) -> int:
        n_row = len(dungeon)
        n_col = len(dungeon[0])
        # dp[i][j]走到右下角需要的最小的血量
        dp = [[0 for _ in range(n_col)] for _ in range(n_row)]
        if dungeon[n_row - 1][n_col - 1] >= 0:
            # 最少也需要1滴血
            dp[n_row - 1][n_col - 1] = 1
        else:
            # 如果最后一个格子需要扣血
            # 最后一个格子扣的血加1
            dp[n_row - 1][n_col - 1] = 1 - dungeon[n_row - 1][n_col - 1]

        for row in range(n_row - 1, -1, -1):
            for col in range(n_col - 1, -1, -1):
                if row == n_row - 1 and col == n_col - 1:
                    # 如果从右下角的格子出发
                    continue
                if row == n_row - 1:
                    # 如果从最后一行出发
                    dp[row][col] = dp[row][col + 1] - dungeon[row][col]

                elif col == n_col - 1:
                    # 如果从最后一列的某个位置出发
                    dp[row][col] = dp[row + 1][col] - dungeon[row][col]

                else:
                    # 从上走过来或者从左边走过来，选择扣血最少的路走过来
                    dp[row][col] = min(dp[row + 1][col] - dungeon[row][col], dp[row][col + 1] - dungeon[row][col])

                if dp[row][col] < 1:
                    dp[row][col] = 1
        return dp[0][0]


if __name__ == '__main__':
    s = Solution()
    res = s.calculateMinimumHP([[1, -4, 5, -99], [2, -2, -2, -1]])
    print(res)
