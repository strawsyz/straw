from typing import *

# 使用哈希列表


class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        hashmap = {}
        for i, num in enumerate(nums):
            res = hashmap.get(target - num)
            if res is not None:
                return [res, i]
            hashmap[num] = i


if __name__ == '__main__':
    nums = [2, 7, 11, 15]
    target = 9
    s = Solution()
    res = s.twoSum(nums, target)
    print(res)