# 还是使用滑动窗口
# 比上一版
# 增加了初始化时，移动到s中第一个必要字符

class Solution:
    def minWindow(self, s: str, t: str) -> str:
        # have和need必须分开幅值，否则会用同一个指针
        have, need = {}, {}
        # left指向第一个需要的字符
        left = right = match = 0
        for c in t:
            need[c] = need.get(c, 0) + 1
        # 初始的最小长度要比s字符串长
        # 因为有可能s和t是相同的
        min_length = len(s) + 1
        # 默认找不到匹配的结果
        res = ""
        t_set = set(t)
        max_match = len(t_set)
        # 找到s中第一个需要的字符
        for c in s:
            if c not in t_set:
                right += 1
                left += 1
            else:
                break

        for right in range(right, len(s)):
            c = s[right]
            if c in t_set:
                have[c] = have.get(c, 0) + 1
                if have[c] == need[c]:
                    match += 1
                # 不断将左指针向有移动，直到match小于max_match
                while match == max_match:
                    # 如果满足所有需要的自字符了
                    # 判断当前是否是最小的长度
                    if min_length > right - left:
                        # 如果是，就更新最小长度
                        min_length = right - left
                        res = s[left: right + 1]
                    if s[left] in t_set:
                        have[s[left]] = have.get(s[left]) - 1
                        if have[s[left]] < need[s[left]]:
                            match -= 1
                    # 左指针向右移动一位
                    left += 1
        return res
