# The size of the input 2D-array will be between 3 and 1000.
# Every integer represented in the 2D-array will be between 1 and N, where N is the size of the input array.

# 使用了查并集算法
# 任何一条边的两个点，如果在某个位置连载一起说明，应为这条边，而产生了环

# 找到有相同根的两个节点
class Solution:
    def findRedundantConnection(self, edges: List[List[int]]) -> List[int]:
        # temp的y位置上的数值表示连接到y的点 【0,0,1,1】表示从1出发，连接到点2和3,第一个0是摆饰，node从1开始
        temp = [0] * 1001
        # 用查并集
        for edge in edges:
            # x 是edge【0】的最头部点
            x = self.find(edge[0], temp)
            y = self.find(edge[1], temp)
            if x != y:
                temp[y] = x
            else:
                return edge

    def find(self, node, pre):
        # 从已经保存的点中寻找没有连接到node
        while pre[node] != 0:
            # 不等于0说明已经不是初始化之后的状态了，已经被改变过了一次了
            # 从node点出发，向头寻找
            node = pre[node]
        # 将node的祖宗点作为返回值
        return node
