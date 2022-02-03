
# 感觉逻辑没问题，但是会报错
# 下面这个用例无法通过
# "pneumonoultramicroscopicsilicovolcanoconiosis"
# "ultramicroscopically"

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
        dp = [[0 for _ in range(n_col)] for _ in range(n_row)]

        if word1[0] != word2[0]:
            # 如果word1的第一个字符和word2的第一个字符不相同
            # 则word1的第一个字符变成Word2的第一个字符需要一次变化
            dp[0][0] = 1
            for row in range(1, n_row):
                if word1[row] == word2[0]:
                    dp[row][0] = dp[row - 1][0]
                else:
                    dp[row][0] = dp[row - 1][0] + 1
            for col in range(1, n_col):
                if word1[0] == word2[col]:
                    dp[0][col] = dp[0][col - 1]
                else:
                    dp[0][col] = dp[0][col - 1] + 1
        else:
            dp[0][0] = 0
            for row in range(1, n_row):
                dp[row][0] = row
            for col in range(1, n_col):
                dp[0][col] = col
        for row in range(1, n_row):
            for col in range(1, n_col):
                if word1[row] == word2[col]:
                    dp[row][col] = dp[row - 1][col - 1]
                else:
                    dp[row][col] = min(dp[row - 1][col - 1], dp[row][col - 1], dp[row - 1][col]) + 1
        return dp[n_row - 1][n_col - 1]
