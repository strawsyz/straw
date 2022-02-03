# 找到用给定的字符串中的数字所能组成最长的回文

class Solution:
    def longestPalindrome(self, s: str) -> int:
        # 根据拥有每个字符的数量保存每个字符
        chars = {}
        for c in s:
            chars[c] = chars.get(c, 0) + 1
        res = 0
        # 如果有单数的字符，最后结果要加上1
        plus_one_flag = False
        for value in chars.values():
            temp = value % 2
            if temp:
                # 如果某个字符的数量是单数
                # 将flag设为True
                plus_one_flag = True
            res += (value - temp)
        if plus_one_flag:
            return res + 1
        else:
            return res
