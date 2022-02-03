class Solution:
    def isBipartite(self, graph: List[List[int]]) -> bool:
        # 每个节点所属的派系
        # 派系分为0和1，-1表示还没有访问过
        v = [-1] * len(graph)
        # 设置0节点的派系为0
        v[0] = 0
        # 待访问的队列
        queue = [0]
        count = 1
        # 最后一个点连接的信息，其实在前面的点中都表达了，
        # 所以count < len(v)
        while count < len(v):
            if not queue:
                # 找到第一个-1的下标，即找到第一个还没有访问的点
                x = v.index(-1)
                queue.append(x)
                v[x] = 0
                count += 1
            while queue:
                p = queue.pop(0)
                for s in graph[p]:
                    if v[s] == -1:
                        # 还没有被访问
                        queue.append(s)
                        # 有s和p连接着所以将他们分为不同的派系
                        # 因为派系分为0和1，所以用下面的公式就能改变派系
                        v[s] = 1 - v[p]
                        count += 1
                    else:
                        # 已经被访问过了
                        if v[s] == v[p]:
                            return False
        return True
