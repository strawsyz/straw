# 使用字典

class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        dict = {}
        for num in nums:
            if num in dict:
                dict.pop(num)
            else:
                dict[num] = 1
        for i in dict.keys():
            return dict[i]