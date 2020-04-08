# Given two strings s and t , write a function to determine if t is an anagram of s.

# 效果比上一版差
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        # 使用字典来存储字符串，每个字符的数量。
        dict = {}
        # 循环字符串，构造数组
        for c in s:
            dict[c] = dict.get(c, 0) + 1
        # 遍历字符串，修改数组
        for c in t:
            dict[c] = dict.get(c, 0) - 1
        for i in dict.values():
            if i != 0:
                return False
        return True
