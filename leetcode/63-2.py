from typing import List


# 动态规划法

# 还可以改进空间复杂度
# 将obstacleGrid当做dp来存储，就可以剩下dp的空间
class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        n = len(obstacleGrid)  # n_row
        m = len(obstacleGrid[0])  # n_col
        # dp[i][j]存储从点[n-i-1,m-j-1]到右下角的点能走的路径数量
        dp = [[0 for _ in range(m)] for _ in range(n)]

        for col in range(m):
            for row in range(n):
                if obstacleGrid[n - row - 1][m - col - 1] == 1:
                    # 如果当前格子有障碍物
                    # 就把路径数设为0
                    dp[row][col] = 0
                elif col == 0:
                    # 从最后一列的某个格子走到右下
                    if row == 0:
                        # 如果是最后一个格子，设置为有一条路径
                        dp[row][col] = 1
                    else:
                        # 如果中间有一个障碍物，路径就是0
                        dp[row][col] = min(1, dp[row - 1][col])
                elif row == 0:
                    # 从最后一行的某个格子走到右下
                    # 如果中间有一个障碍物，路径就是0
                    dp[row][col] = min(1, dp[row][col - 1])
                else:
                    # 当前格子数，等于下面的路径数和左边点的路径数之和
                    dp[row][col] = dp[row - 1][col] + dp[row][col - 1]
        return dp[n - 1][m - 1]


if __name__ == '__main__':
    s = Solution()
    res = s.uniquePathsWithObstacles([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    print(res)
