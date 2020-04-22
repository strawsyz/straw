# 照着官方答案写了一遍

class Solution:
    def uniquePathsIII(self, grid):
        n_row, n_col = len(grid), len(grid[0])

        def neighbors(r, c):
            # 找到不超过边界的点
            for nr, nc in ((r - 1, c), (r, c - 1), (r + 1, c), (r, c + 1)):
                if 0 <= nr < n_row and 0 <= nc < n_col and grid[nr][nc] % 2 == 0:
                    yield nr, nc

        # 需要走过的空格的数量
        todo = 0
        for r, row in enumerate(grid):
            for c, val in enumerate(row):
                if val != -1: todo += 1
                if val == 1: sr, sc = r, c  # 开始的点
                if val == 2: tr, tc = r, c  # 结束的点

        self.ans = 0

        def dfs(r, c, todo):
            # 进入新的格子，较少一个需要走的格子
            todo -= 1
            if todo < 0: return
            if r == tr and c == tc:
                # 走到了最后的点
                if todo == 0:
                    # 走过了所有的点
                    self.ans += 1
                return
            # 走过的格子设为-1防止之后再走
            grid[r][c] = -1
            for nr, nc in neighbors(r, c):
                # 筛选出能够走的点
                dfs(nr, nc, todo)
            # 回退一步，把当前的格子设为没有走过
            grid[r][c] = 0

        dfs(sr, sc, todo)
        return self.ans
