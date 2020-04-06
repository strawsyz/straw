def f(number):
    memo = [-1] * (number + 1)

    def dp(n):
        if n == 0 or n == 1:
            return 1
        if memo[n] == -1:
            tmp = 0
            for i in range(1, n + 1):
                temp += dp(n - i)
            memo[n] = tmp
        return memo[n]

    return dp(n)
