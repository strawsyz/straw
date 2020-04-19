# 位运算的应用
# 相同的数字异或运算之后会变成0
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        res = 0
        for num in nums:
            res ^= num
        return res
