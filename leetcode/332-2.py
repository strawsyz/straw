from typing import List
import collections


# 使用深度遍历
# 有向图

# 考虑到了没有合理路径的情况
# 先判断是否有合理路径
# 没有的话，返回False
# 如果有的话，就用深度遍历查找

# 判断是否有合理路径 一笔画问题  欧拉环路

# 虽然题目写了一定有合理路径。但我还是写了判断是否合理路径
# 明明添加了一堆无用的判断代码，不知道为什么结果速度要快很多


class Solution:
    def findItinerary(self, tickets: List[List[str]]) -> List[str]:
        def temp():
            return [0, 0]

        # 记录[入度，出度]
        # in_out_map[key] = [0, 0]
        in_out_map = collections.defaultdict(temp)

        # 添加新的键值对的时候，默认值为空列表
        # 存储每个结点之间的所能连接到的位置
        d = collections.defaultdict(list)
        for from_, to in tickets:
            if from_ != to:
                d[from_] += [to]  # 路径存进邻接表
                # 设置入度和出度
                in_out_map[from_][1] += 1
                in_out_map[to][0] += 1

        # 默认从JFK出发
        start = 'JFK'
        in_more_than_out = 0
        out_more_than_in = 0
        for key, in_out in in_out_map.items():
            in_ = in_out[0]
            out = in_out[1]
            if in_ - out == 0:
                continue
            elif in_ - out < -1 or in_ - out > 1:
                return False
            elif in_ - out == 1:
                in_more_than_out += 1
                if in_more_than_out > 1:
                    return False
            elif in_ - out == -1:
                # 入度大于出度的点作为start点
                start = key
                out_more_than_in += 1
                if out_more_than_in > 1:
                    return False
        if in_more_than_out - out_more_than_in != 0:
            return False

        for from_ in d:
            d[from_].sort()  # 邻接表排序
        ans = []

        def dfs(f):  # 深搜函数
            while d[f]:
                dfs(d[f].pop(0))  # 路径检索
            ans.insert(0, f)  # 放在最前

        dfs(start)
        return ans


if __name__ == '__main__':
    s = Solution()
    res = s.findItinerary([["MUC", "LHR"], ["JFK", "MUC"], ["SFO", "SJC"], ["LHR", "SFO"]])
    print(res)
