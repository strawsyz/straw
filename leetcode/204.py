# 找小于n的质数的个数
# 理论上是正确的
# 但是会超时，但是空间复杂度低
class Solution:
    def countPrimes(self, n: int) -> int:
        if n < 3:
            return 0
        results = [2]
        for i in range(3, n, 2):
            is_prime = True
            for prime in results:
                if i % prime == 0:
                    is_prime = False
                    break
            if is_prime:
                results.append(i)
        return len(results)
