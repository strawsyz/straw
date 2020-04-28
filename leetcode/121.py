from typing import List


class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if prices == []:
            return
        max_profit = 0
        max = prices[-1]
        for i in range(len(prices) - 2, -1, -1):
            if prices[i] < max:
                if max_profit < max - prices[i]:
                    max_profit = max - prices[i]
            elif prices[i] > max:
                max = prices[i]
        return max_profit


if __name__ == '__main__':
    print(Solution().maxProfit([7, 1, 5, 3, 6, 4]))
