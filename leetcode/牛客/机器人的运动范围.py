# -*- coding:utf-8 -*-

# 深度遍历
class Solution:
    def movingCount(self, threshold, rows, cols):
        # 题目要求threshold小于0的时候返回0
        if threshold <= 0:
            return 0
        # write code here
        moves = [[1, 0], [0, 1], [-1, 0], [0, -1]]
        counter = 0
        visited = [[0 for _ in range(cols)] for _ in range(rows)]

        def is_ok(row, col, n_row, n_col, k=threshold):
            if row < 0 or row >= n_row:
                return False
            if col < 0 or col >= n_col:
                return False
            sum = 0
            while row > 0:
                sum += row % 10
                row = row // 10
            while col > 0:
                sum += col % 10
                col = col // 10
            return sum <= k

        start = [0, 0]
        counter += 1
        visited[0][0] = 1

        def dfs(now, n_row, n_col):
            counter = 0
            for move in moves:
                next_step = [now[0] + move[0], now[1] + move[1]]
                if is_ok(next_step[0], next_step[1], n_row, n_col):
                    if visited[next_step[0]][next_step[1]] == 0:
                        visited[next_step[0]][next_step[1]] = 1
                        counter += 1
                        counter += dfs(next_step, n_row, n_col)
            return counter

        return dfs(start, rows, cols) + 1


if __name__ == '__main__':
    s = Solution()
    res = s.movingCount(2, 6, 2)
    print(res)
