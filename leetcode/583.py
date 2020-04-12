class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        length_1 = len(word1)
        length_2 = len(word2)
        # dp[i][j]是word1前i个字符和word2的前j个字符最长相同的字串
        dp = [[None for _ in range(length_2 + 1)] for _ in range(length_1 + 1)]
        # 先用动态规划法求出word1前i个字符和word2的前j个字符最长相同的字串
        for row in range(length_1):
            for col in range(length_2):
                if row == 0 or col == 0:
                    dp[row][col] = 0
                    continue
                if word1[row - 1] == word2[col - 1]:
                    dp[row][col] = dp[row - 1][col - 1] + 1
                else:
                    dp[row][col] = max(dp[row - 1][col], dp[row][col - 1])
        return length_1 + length_1 - 2 * dp[length_1][length_2]
