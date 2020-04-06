

class Solution:

    def minWindow(self, s: str, t: str) -> str:

        left, right, match = 0, 0, 0
        need, have = {}, {}
        result = ""
        len_s = len(s)
        result_len = len_s + 1
        for x in t:
            # key表示存储的字符，value表示字符的个数
            need[x] = need.get(x, 0) + 1
        tset = set(t)
        match_len = len(tset)
        while right < len_s:
            if s[right] in tset:
                have[s[right]] = have.get(s[right], 0) + 1
                if have[s[right]] == need[s[right]]:
                    match += 1
            right += 1
            while match == match_len:
                if right - left < result_len:
                    result = s[left:right]
                    result_len = len(result)
                if s[left] in tset:
                    have[s[left]] = have.get(s[left], 0) - 1
                    if have[s[left]] < need[s[left]]:
                        match -= 1
                left += 1

        return result
