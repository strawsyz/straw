from typing import List


class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        left = right = match = 0
        need, have = {}, {}
        for char in p:
            need[char] = need.get(char, 0) + 1
        p_set = set(p)
        p_length = len(p)
        result = []
        need_match = len(p_set)

        for char in s:
            if char in p_set:
                have[char] = have.get(char, 0) + 1
                if have[char] == need[char]:
                    match += 1

            right += 1
            while match == need_match:
                if right - left == p_length:
                    result.append(left)
                if s[left] in p_set:
                    have[s[left]] = have.get(s[left]) - 1
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
