# 稍微有点绕远路
# 空间复杂度还可以优化
# 解法：
# 找到一个非负数，作为第一值
# 把原数组中的连续的正数整理到一个一个正数中
# 把原数组中的连续的负数和0整理到一个一个负数中
# 然后用动态规划（Kadane 算法）


class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        # pre = 1  # 1表示上一个数字是大于等于0，0表示小于0
        if len(nums) == 0:
            return 0
        start = len(nums)

        # 找到第一个非负数
        for i, num in enumerate(nums):
            if num > 0:
                start = i
                break
        if start == len(nums):
            # 如果真个数组都是负数
            return max(nums)
        pre = 1
        new_nums = []
        temp_sum = nums[start]
        for num in nums[start + 1:]:
            if num > 0:
                if pre == 1:
                    temp_sum += num
                else:
                    new_nums.append(temp_sum)
                    temp_sum = num
                    pre = 1
            else:
                if pre == 0:
                    temp_sum += num
                else:
                    new_nums.append(temp_sum)
                    temp_sum = num
                    pre = 0
        new_nums.append(temp_sum)
        sum = 0
        max_sum = 0
        for num in new_nums:
            if num > 0:
                sum += num
                if sum > max_sum:
                    max_sum = sum
            else:
                sum += num
                # 如果前面的数字加起来小于0就重置
                if sum < 0:
                    sum = 0

        return max_sum
