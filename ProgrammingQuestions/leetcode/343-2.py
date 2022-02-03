# 理论上来讲速度变快了


class Solution:
    def integerBreak(self, n: int) -> int:
        if n == 2:
            return 1
        elif n == 3:
            return 2
        res = [0 for _ in range(max(n + 1, 7))]
        res[2] = 2
        res[3] = 3
        if n > 3:
            for i in range(4, n + 1):
                for j in range(1, i // 2 + 1):
                    res[i] = max(res[i], res[j] * res[i - j])
        return res[n]


if __name__ == '__main__':
    n = 3
    res = Solution().integerBreak(n)
    print(res)
