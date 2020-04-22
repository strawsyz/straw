# 一次比较每个字符串的第i个数字
# 可以优化为，两两比较

class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        # 初始化为很大的值
        min_length = 10 ** 10
        if len(strs) == 1:
            # 如果只有一个字符串，就直接返回它
            return strs[0]
        elif len(strs) == 0:
            # 如果是空列表
            return ""
        # 找到最短的长度
        for i in strs:
            if min_length > len(i):
                min_length = len(i)
        if min_length == 0:
            # 如果最短长度为0 ，就返回空字符串
            return ""
        for i in range(0, min_length):
            pre = strs[0][i]
            # 循环每个字符串的第i个数字
            for chars in strs:
                # 如果遇到了一个不同的数字
                if pre != chars[i]:
                    return strs[0][:i]
        # 如果所有的字符串都完全相同
        # 就返回任意一个字符串的前min_length个字符
        return strs[0][:min_length]

