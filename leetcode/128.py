# 使用官方的解法
# 虽然for 中有一个while循环
# 但是while循环只有在一定条件下才发生
# 所以看做时间复杂度O（n）
class Solution:
    def longestConsecutive(self, nums):
        longest_streak = 0
        # 设置为set。防止相同的数字产生干扰
        num_set = set(nums)

        for num in num_set:
            if num - 1 not in num_set:
                # 如果无法找到num-1.说明num可以作为一个序列的起始位置
                current_num = num
                current_streak = 1
                # 已num-1作为起点，计算最长的序列
                while current_num + 1 in num_set:
                    current_num += 1
                    current_streak += 1

                longest_streak = max(longest_streak, current_streak)

        return longest_streak
