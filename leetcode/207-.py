from typing import List


# dfs,会超时

class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        map = {}
        for i in prerequisites:
            res = map.get(i[0])
            if res is None:
                temp = set()
                temp.add(i[1])
                map[i[0]] = temp
            else:
                res.add(i[1])
                map[i[0]] = res

        # 返回True表示不成环
        def dfs(haved, key):
            res = True
            is_have = map.get(key)
            if is_have is None:
                return True
            for value in is_have:
                if value in haved:
                    # 成环
                    return False
                else:
                    haved.append(value)
                    # 将成环的结果向上传递
                    res = res and dfs(haved, value)
                    haved.pop()
            return res

        for key in map.keys():
            # 当返回False的时候表示，内部成环了，说明无法读完了
            if not dfs([key], key):
                return False
        return True


if __name__ == '__main__':
    s = Solution()
    res = s.canFinish(2, [[1, 0]])
    print(res)
