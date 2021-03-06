# 记忆化搜索是自上而下
# 动态规划是自下而上（比较好）
# 使用了动态规划法


class Solution:
    def climbStairs(self, n: int) -> int:
        memo = [-1] * (n + 1)
        for i in range(0, n + 1):
            if i == 0 or i == 1:
                memo[i] = 1
            else:
                memo[i] = memo[i - 1] + memo[i - 2]
        return memo[n]


