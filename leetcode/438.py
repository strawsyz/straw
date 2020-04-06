import copy
from typing import List


class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        left = right = match = 0
        need, have = {}, {}

        for i in p:
            need[i] = need.get(i, 0) + 1
        p_set = set(p)
        p_set_len = len(p_set)
        p_len = len(p)
        result = []

        for cha in s:
            if cha in p_set:
                have[cha] = have.get(cha, 0) + 1
                if have[cha] == need[cha]:
                    match += 1
            right += 1
            while match == p_set_len:
                if right - left == p_len:
                    # 当match等于需要的数值，同时窗口的长度等于p的长度时
                    result.append(left)
                if s[left] in p_set:
                    have[s[left]] = have.get(s[left], 0) - 1
                    if have[s[left]] < need[s[left]]:
                        match -= 1
                left += 1

        return result



if __name__ == '__main__':
    s = Solution()
    str = 'cbaebabacd'
    p = 'abc'
    res = s.findAnagrams(str, p)
    print(res)
