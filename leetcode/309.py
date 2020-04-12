from typing import List
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        n = len(prices)
        if n == 0 or n == 1:
            return 0
        # 第i天拥有股票的最大profit
        hold = [None for _ in range(n)]
        # 第i天卖出股票的最大profit
        unhold = [None for _ in range(n)]
        hold[0] = -prices[0]
        hold[1] = max(-prices[0], -prices[1])
        unhold[0] = 0
        for i in range(1, n):
            if i > 1:
                # 比较第i天买入和第i天不买入的利益
                # 第i天买入的利益，等于第i-2天卖出的最大利益减去第i天的股价
                hold[i] = max(hold[i - 1], unhold[i - 2] - prices[i])
            # 比较第i天卖出与不卖出的收益
            # 第i天卖出的最大利益等于第i-1天持有股票状态下的最大利益加上第i天的股价
            unhold[i] = max(unhold[i - 1], hold[i - 1] + prices[i])
        return unhold[n - 1]


s = Solution()
res = s.maxProfit([1,2,3,0,2])
print(res)