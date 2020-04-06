# 给定一个有序整数型和一个整数，在其中找两个元素，使其和为target。返回两个数的索引
# 说明:
# 返回的下标值（index1 和 index2）不是从零开始的。
# 你可以假设每个输入只对应唯一的答案，而且你不可以重复使用相同的元素。
# 本提使用对撞指针
from typing import *


class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        left = 0
        right = len(numbers) - 1
        while right > left:
            min = numbers[left]
            while numbers[right] > target - min:
                right -= 1
            # left += 1
            max = numbers[right]
            while max < target - numbers[left]:
                left += 1

            if numbers[right] + min == target:
                return [left+1, right+1]

        return []