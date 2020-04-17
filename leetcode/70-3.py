# 还是动态规划，从最后一个台阶向前推
# 每次只存储前两个台阶的情况，根据前两个台阶的情况算出当前的情况
# 理论上能减少空间复杂度

class Solution:
    def climbStairs(self, n: int) -> int:
        # 特殊情况处理
        if n < 4:
            return n
        pre_pre = 1
        pre = 2
        for i in range(2, n):
            res = pre + pre_pre
            pre_pre = pre
            pre = res
        return res
