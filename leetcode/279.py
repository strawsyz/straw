# 求最短路径，广度优先搜索法（BFS）

class Node:
    # 使用Node来存储val和passed参数
    def __init__(self, x, passed=0):
        self.val = x
        # 从目标数字出发到值为val的节点的最短路径（需要走几步）
        self.passed = passed


class Solution:

    def numSquares(self, n: int) -> int:
        # 建立一个队列，用于存储Node
        queue = [Node(n)]
        # 用数组表示节点是否被访问过了，从0开始计算，第n+1的节点（即Node（n））默认已经被访问过了
        visited = [0] * n + [1]
        while queue:
            # 队列先进先出。所以拿出列表中第一个Node
            x = queue.pop(0)
            i = 1
            while True:
                temp = x.val - i * i
                if temp < 0:
                    break
                if temp == 0:
                    return x.passed + 1
                # 判断当前的点是否已经被访问过了，
                if not visited[temp]:
                    # 如果当前节点没有被访问过
                    queue.append(Node(temp, x.passed + 1))
                    # 标记已经访问过了
                    visited[temp] = 1
                i += 1


if __name__ == '__main__':
    s = Solution()
    res = s.numSquares(12)
    print(res)
