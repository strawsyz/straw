# Given a non-empty 2D array grid of 0's and 1's, an island is a group of 1's (representing land)
# connected 4-directionally (horizontal or vertical.) You may assume all four edges of the grid
# are surrounded by water.

# Find the maximum area of an island in the given 2D array.
# (If there is no island, the maximum area is 0.)

# DFS常用于解决可达性问题

class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        # 特殊情况处理
        if len(grid) == 0:
            return 0

        def dfs(i, j, grid, visited, area):
            # 判断是否还在盘面上
            if i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]):
                return
            # 如果已经访问过或者不是陆地
            if grid[i][j] == 0 or visited[i][j] == 1:
                return
            visited[i][j] = 1
            area[0] += 1
            # 访问四面
            dfs(i - 1, j, grid, visited, area)
            dfs(i + 1, j, grid, visited, area)
            dfs(i, j - 1, grid, visited, area)
            dfs(i, j + 1, grid, visited, area)

        # 获得盘面的行数和列数
        n_row, n_column = len(grid), len(grid[0])
        visited = [[0] * row for _ in range(n_column)]
        # 存储大于最大面积
        max_area = 0
        for row in range(n_row):
            for column in range(n_column):
                # 如果当前位置还没有访问过，而且是个陆地
                if visited[row][column] == 0 and grid[row][column] == 1:
                    # 因为area要作为递归函数的参数传进去，所以要设置为数组
                    # area 存储当前岛屿的面积
                    area = [0]
                    # 用dfs搜索当前岛屿面积
                    dfs(row, column, grid, visited, area)
                    if max_area < area[0]:
                        max_area = area[0]
        return max_area
