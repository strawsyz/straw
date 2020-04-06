memo = [-1] * (n - 1)


# 使用空间来换取时间

def f(n):
    if n < 2:
        return n
    if memo[n] == -1:
        memo[n] = f([n - 1]) - f([n - 2])
    return memo[n]
