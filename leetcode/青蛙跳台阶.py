def f(n):
    memo = [-1] * (n + 1)
    def dp(n):
        if n==1 or n==0:
            return 1
        if memo[n]==-1:
            memo[n] = (dp(n-1) + dp(n-2))% 1000000007
        return memo
    return dp(n)
