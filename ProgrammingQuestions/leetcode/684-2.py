from typing import List
# The size of the input 2D-array will be between 3 and 1000.
# Every integer represented in the 2D-array will be between 1 and N, where N is the size of the input array.

# 使用了查并集算法
# 任何一条边的两个点，如果在某个位置连载一起说明，这条边是附加边，而产生了环
# 同样是查并集算法，但速度要慢很多

class UnionFind():
    def __init__(self, n):
        # 存储每个下标对应的派系
        self.pre = []
        # 初始化为每个下标自成一派
        # 注意，下标看做从1开始
        for i in range(n + 1):
            self.pre.append(i)

    def union(self, u, v):
        # 要将u和v两个位置连在一起
        u_id = self.find(u)
        v_id = self.find(v)
        # 如果u和v的上级不相同，说明两个还不属于同一个派系
        if u_id == v_id:
            return
        # 遍历u派系的所有人，让所有人以后跟v派系的人走
        for i in range(len(self.pre)):
            # 找到u派系的人
            if self.pre[i] == u_id:
                # 将派系改为v
                self.pre[i] = v_id

    def find(self, p):
        return self.pre[p]

    def connect(self, u, v):
        # 检测u和v是否属于同一个派系
        return self.find(u) == self.find(v)


class Solution:
    def findRedundantConnection(self, edges: List[List[int]]) -> List[int]:
        uf = UnionFind(len(edges))
        for edge in edges:
            u, v = edge[0], edge[1]
            # 如果新来的边的加入，并没有改变派系关系，
            # 即，将本来就连通的两个点，重新连了一遍
            if uf.connect(u, v):
                return u, v
            else:
                uf.union(u, v)
        return -1, -1


if __name__ == '__main__':
    input = [[1, 2], [1, 3], [2, 3]]
    s = Solution()
    res = s.findRedundantConnection(input)
    print(res)
