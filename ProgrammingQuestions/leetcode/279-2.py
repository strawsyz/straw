from math import sqrt


# 使用动态规划的方法解决问题
# 比前一种方法更加耗时

class Solution:

    def numSquares(self, n: int) -> int:

        dp = [0] * (n + 1)
        dp[1] = 1
        square_nums = [1]
        for i in range(2, n + 1):
            if int(sqrt(i)) == sqrt(i):
                square_nums.append(i)
                dp[i] = 1
                continue
            min = 10 ** 10
            for square_num in square_nums:
                if min > dp[i - square_num]:
                    min = dp[i - square_num]
            dp[i] = min + 1
        return dp[n]


if __name__ == '__main__':
    s = Solution()
    res = s.numSquares(12)
    print(res)
