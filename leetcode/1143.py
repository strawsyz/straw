# lcs问题
# 最长公共子序列
# 时间复杂度 mxn
# 空间复杂度 mxn


class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        if text1 == text2:
            return len(text1)
        if text1 == '' or text2 == '':
            return 0
        dp = [[0] * (len(text2) + 1) for _ in range(len(text1) + 1)]
        for row in range(1, len(text1) + 1):
            for col in range(1, len(text2) + 1):
                if text1[row-1] == text2[col-1]:
                    dp[row][col] = dp[row - 1][col - 1]+1
                else:
                    dp[row][col] = max(dp[row - 1][col], dp[row][col - 1])
        return dp[-1][-1]
