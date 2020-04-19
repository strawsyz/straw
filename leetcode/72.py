# 使用动态规划法


class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        # 两个字符串的长度要一职
        # 尽量让两个字符串中大部分的字符顺序一致
        # 尽量用替代的手法，而不是用删除加添加
        # 所以要让两个字符串，相同字符的相对位置尽量一致
        # 比如 abc  adc ，只需要一个动作
        # ace adc要两个动作
        # word1的长度作为行数
        n_row = len(word1)
        # word2的长度作为列数
        n_col = len(word2)
        # 如果word1长度为0
        if n_row == 0:
            # 返回word2的长度
            return n_col
        # 如果word2长度为0
        if n_col == 0:
            # 返回word1的长度
            return n_row
        # dp[i][j]表示word1前i个字符转化word2前j个字符需要变化的次数
        dp = [[0 for _ in range(n_col + 1)] for _ in range(n_row + 1)]
        # 要包含第n_row个字符，所以要加1
        for row in range(1, n_row+1):
            # 字符串变为空字符串需要字符串长度个操作
            dp[row][0] = row
        for col in range(1, n_col+1):
            # 空字符串变为字符串需要字符串长度个操作
            dp[0][col] = col

        for row in range(1, n_row+1):
            for col in range(1, n_col+1):
                if word1[row-1] == word2[col-1]:
                    # 如果相同，等于两个字符串分别减去一个字符的时候操作数
                    dp[row][col] = dp[row - 1][col - 1]
                else:
                    # 如果不相同，之前的最少变化数加一
                    dp[row][col] = min(dp[row - 1][col - 1], dp[row][col - 1], dp[row - 1][col]) + 1
        return dp[n_row][n_col]
