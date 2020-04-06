class Node:
    def __init__(self, x, passed=0):
        self.val = x
        # 最短路径需要走的步数
        self.passed = passed


class Solution:

    def numSquares(self, n: int) -> int:
        queue = [Node(n)]
        visited = [0] * n + [1]
        while queue:
            # 先进先出，因为先进的是比较靠进n的节点，所以要先处理！
            x = queue.pop(0)
            i = 1
            while True:
                temp = x.val - i * i
                if temp < 0:
                    break
                if temp == 0:
                    return x.passed + 1
                if not visited[temp]:
                    queue.append(Node(temp, x.passed + 1))
                    visited[temp] = 1
                i += 1

if __name__ == '__main__':
    s = Solution()
    res = s.numSquares(12)
    print(res)