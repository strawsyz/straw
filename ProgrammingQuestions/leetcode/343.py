class Solution:
    def integerBreak(self, n: int) -> int:
        res = [0 for i in range(n + 1)]
        res[2] = 1
        for i in range(3, n + 1):
            for j in range(1, i // 2 + 1):
                res[i] = max(res[i], max(res[j], j) * max(res[i - j], i - j))
            # res[i] = temp_max
        return res[n]


#
#
if __name__ == '__main__':
    n = 3

    res = Solution().integerBreak(n)
    print(res)
