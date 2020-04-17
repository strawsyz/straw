# 还是动态规划，代码更加简洁

class Solution:
    def climbStairs(self, n: int) -> int:
        # 特殊情况处理
        if n < 4:
            return n

        # 存储从倒数第i+2个到最后一个台阶的走法
        dp = [0 for _ in range(n)]
        dp[0] = 1
        dp[1] = 2
        for i in range(2, n):
            dp[i] = dp[i - 1] + dp[i - 2]
        return dp[n - 1]
