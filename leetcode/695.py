# Given a non-empty 2D array grid of 0's and 1's, an island is a group of 1's (representing land)
# connected 4-directionally (horizontal or vertical.) You may assume all four edges of the grid
# are surrounded by water.

# Find the maximum area of an island in the given 2D array.
# (If there is no island, the maximum area is 0.)

class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:

        def dfs(i, j, grid, visited, area):
            if i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]):
                return
            if grid[i][j] == 0 or visited[i][j] == 1:
                return
            visited[i][j] = 1
            area[0] += 1
            dfs(i - 1, j, grid, visited, area)
            dfs(i + 1, j, grid, visited, area)
            dfs(i, j - 1, grid, visited, area)
            dfs(i, j + 1, grid, visited, area)


        if len(grid) == 0:
            return 0
        row, column = len(grid[0]), len(grid)
        visited = [[0] * row for _ in range(column)]
        max_area = 0
        for i in range(column):
            for j in range(row):
                if visited[i][j] == 0 and grid[i][j] == 1:
                    # 因为area要作为递归函数的参数传进去，所以要设置为数组
                    area = [0]
                    dfs(i, j, grid, visited, area)
                    if max_area < area[0]:
                        max_area = area[0]
        return max_area

