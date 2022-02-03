import collections
from typing import List


class Solution:
    def possibleBipartition(self, N: int, dislikes: List[List[int]]) -> bool:
        # 如果连线数特别多，可以知道不可能分为两个组
        if len(dislikes) > (N // 2 + 1) ** 2:
            return False
        graph = collections.defaultdict(set)
        for u, v in dislikes:
            graph[u].add(v)
            graph[v].add(u)

            # 将上色的点放入集合中
        color = {}

        # 深度遍历
        def dfs(node, c=0):
            """
            :param node: 当前节点编号
            :param c: 连接的节点的颜色
            :return:
            """
            # 如果节点已经上色
            if node in color:
                # 返回上的颜色是否等于c
                return color[node] == c
            # 如果当前的点还没有上色，就给予上色
            color[node] = c
            # 循环所有的邻居节点
            return all(dfs(nei, c ^ 1) for nei in graph[node])

        for node in range(1, N + 1):
            # 循环所有的节点
            if node not in color:
                # 检查没上色的点的邻居节点的情况
                if not dfs(node):
                    return False
        return True


if __name__ == '__main__':
    N = 3
    dislikes = [[1, 2], [1, 3], [2, 3]]
    res = Solution().possibleBipartition(N, dislikes)
    print(res)
