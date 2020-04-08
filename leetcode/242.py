# Given two strings s and t , write a function to determine if t is an anagram of s.

class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        # 用一个长度26数组来保存字符串拥有的各类字符的数量
        # 从a来时排序，每个下标对应一个字符。下标对应的数字对应的数字就是改字符的数量
        arr = [0] * 26
        # 循环字符串，构造数组
        ord_a = ord('a')
        for c in s:
            arr[ord(c) - ord_a] += 1
        # 遍历字符串，修改数组
        for c in t:
            arr[ord(c) - ord_a] -= 1
        for i in arr:
            if i != 0:
                return False
        return True
