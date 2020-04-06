class Solution:
    def longestPalindrome(self, s: str) -> int:
        dict = {}
        for char in s:
            if char in dict:
                dict[char] += 1
            else:
                dict[char] = 1
        res = 0
        for char in dict.keys():
            value = dict.get(char)
            res += (value // 2) * 2
        if len(s) > res:
            res += 1
        return res
