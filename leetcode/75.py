# 解法：先计算0,1,2的个数
# 根据0,1,2的个数设置nums的值

class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        count_0 = 0
        count_1 = 0
        count_2 = 0
        for i in nums:
            if i == 0:
                count_0 += 1
            elif i == 1:
                count_1 += 1
            else:
                count_2 += 1
        nums[:count_0] = [0] * count_0
        nums[count_0:count_0 + count_1] = [1] * count_1
        nums[count_0 + count_1:count_0 + count_1 + count_2] = [2] * count_2
