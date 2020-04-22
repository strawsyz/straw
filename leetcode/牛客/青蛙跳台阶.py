# 青蛙一次跳1或2格台阶，跳完所有台阶有多少种跳法
# 记忆化搜索
def f(n):

    memo = [-1] * (n + 1)
    def dp(n):
        if n == 1 or n == 0:
            return 1
        if memo[n] == -1:
            memo[n] = (dp(n - 1) + dp(n - 2))
        return memo[n]

    return dp(n)

res = f(5)
print(res)