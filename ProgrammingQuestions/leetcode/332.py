from typing import List
import collections

# 使用深度遍历
# 有向图
class Solution:
    def findItinerary(self, tickets: List[List[str]]) -> List[str]:

        # 添加新的键值对的时候，默认值为空列表
        # 存储每个结点之间的所能连接到的位置
        d = collections.defaultdict(list)
        for f, t in tickets:
            d[f] += [t]  # 路径存进邻接表
        for f in d:
            d[f].sort()  # 邻接表排序
        ans = []

        def dfs(f):  # 深搜函数
            while d[f]:
                dfs(d[f].pop(0))  # 路径检索
            ans.insert(0, f)  # 放在最前

        dfs('JFK')
        return ans

if __name__ == '__main__':
    s = Solution()
    res = s.findItinerary([["MUC","LHR"],["JFK","MUC"],["SFO","SJC"],["LHR","SFO"]])
    print(res)
