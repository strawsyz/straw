# 动态规划法
# 变态青蛙，一次可以跳一格到无穷格

def f(number):
    memo = [-1] * (number + 1)

    def dp(n):
        if n == 0 or n == 1:
            return 1
        if memo[n] == -1:
            temp = 0
            for i in range(1, n + 1):
                temp += dp(n - i)
            memo[n] = temp
        return memo[n]

    return dp(number)
