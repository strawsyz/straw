class Solution:
    def movingCount(self, m: int, n: int, k: int) -> int:
        if k < 0:
            return 0
        moves = [[1, 0], [-1, 0], [0, 1], [0, -1]]
        visited = [[0] * n for _ in range(m)]

        def validate_boundary(x, y, xMax, yMax):
            if x < 0 or x >= xMax:
                return False
            if y < 0 or y >= yMax:
                return False
            return True

        def validate_k(x, y, k):
            res = 0
            while x > 0:
                res += x % 10
                x = x // 10
            while y > 0:
                res += y % 10
                y = y // 10
            return res > k

        def dfs(m, n, x, y, k):
            count = 1
            visited[x][y] = 1
            for i in range(len(moves)):
                newX = x + moves[i][0]
                newY = y + moves[i][1]
                if not validate_boundary(newX, newY, m, n):
                    continue
                if visited[newX][newY] == 1:
                    continue
                if validate_k(newX, newY, k):
                    continue
                count += dfs(m, n, newX, newY, k)
            return count

        return dfs(m, n, 0, 0, k)
