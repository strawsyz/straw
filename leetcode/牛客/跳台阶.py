# -*- coding:utf-8 -*-
class Solution:
    def jumpFloor(self, number):
        if number < 3:
            return number
        # write code here
        dp = [0 for _ in range(number)]
        dp[number - 1] = 1
        dp[number - 2] = 2
        for i in range(number - 3, -1, -1):
            dp[i] = dp[i + 2] + dp[i + 1]
        return dp[0]


if __name__ == '__main__':
    s = Solution()
    res = s.jumpFloor(5)
    print(res)
