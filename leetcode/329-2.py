from typing import List


# 记忆搜索法

class Solution:
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        # 能向4个方向走
        moves = [[0, 1], [1, 0], [0, -1], [-1, 0]]
        # martix是空列表的情况
        if len(matrix) == 0:
            return 0
        n_row = len(matrix)
        n_col = len(matrix[0])
        result = 1
        # 动态规划表，dp[i][j]计入以[i][j]为终点的情况下的最长递增序列的长度
        memo = [[0 for _ in range(n_col)] for _ in range(n_row)]

        def dfs(row, col):
            # 如果当前位置有计算过
            if memo[row][col] != 0:
                # 就直接返回当前位置的
                return memo[row][col]
            for move in moves:
                new_row = row + move[0]
                new_col = col + move[1]
                if new_row >= 0 and new_row < n_row and new_col >= 0 and new_col < n_col and matrix[new_row][new_col] > \
                        matrix[row][col]:
                    # 如果没有超过边界，并且比当前的位置要大
                    memo[row][col] = max(memo[row][col], dfs(new_row, new_col))
            memo[row][col] += 1
            return memo[row][col]

        for row in range(n_row):
            for col in range(n_col):
                result = max(result, dfs(row, col))
        return result


if __name__ == '__main__':
    res = Solution().longestIncreasingPath([[9, 9, 4], [6, 6, 8], [2, 1, 1]])
    print(res)
