# 找小于n的质数的个数

# 参照大佬的代码
class Solution:
    def countPrimes(self, n):
        if n < 3:
            return 0
        # 标志为True的位置标明是一个True
        primes = [True] * n
        primes[0] = primes[1] = False
        # 下面的循环的范围设置得很巧妙
        for i in range(2, int(n ** 0.5) + 1):
            if primes[i]:
                # 将质数的倍数全都设置为False
                primes[i * i: n: i] = [False] * len(primes[i * i: n: i])
        # 将所有True的个数加起来
        return sum(primes)
