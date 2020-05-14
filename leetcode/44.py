# 参考大佬的代码

# 动态规划化法
class Solution:
    def isMatch(self, s, p):
        len_s = len(s)
        # 如果p中的？和字母的数量和大于s的长度
        if len(p) - p.count('*') > len_s:
            return False
        # dp[i]表示字符串s前i个字符是否匹配p
        # dp[0]表示字符串和空字符串或者只有*的字符串匹配
        dp = [True] + [False] * len_s
        for i in p:
            # 一次循环确定：s 与 到字符i位置的p的部分 的拼配气矿
            if i != '*':
                for n in reversed(range(len_s)):
                    # 倒着循环，len_s-1 到 0
                    # 从s的最后一个字符往前遍历
                    if dp[n] and (i in {s[n], "?"}):
                        dp[n + 1] = True
                    else:
                        dp[n + 1] = False
            else:
                for n in range(1, len_s + 1):
                    dp[n] = dp[n - 1] or dp[n]
            dp[0] = dp[0] and i == '*'
        # 返回结果，即s的全部字符是否和模式p匹配
        return dp[-1]


if __name__ == '__main__':
    s = "mississippi"
    p = "m??*ss*?i*pi"
    # s = "adceb"
    # p = "*a*b"
    # s = "abefcdgiescdfimde"
    # p = "ab*cd?i*de"
    # s = "aaaa"
    # p = "***a"
    # s = "c"
    # p = "*?*"
    # s = "hi"
    # p = "*?"

    res = Solution().isMatch(s, p)
    print(res)
