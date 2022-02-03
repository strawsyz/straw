from typing import *


# 使用哈希列表


class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        # 因为本道题目确保有解，所以不用考虑特殊情况

        # 使用哈希表来存储需要的数字，key表示数字，value表示在nums中的索引
        hashmap = {}

        for i, num in enumerate(nums):
            res = hashmap.get(target - num)
            # 不能直接使用res作为判断条件，因为res有可能是0
            # 如果res为0，就会把它当做None处理
            if res is not None:
                return [res, i]
            else:
                hashmap[num] = i


if __name__ == '__main__':
    nums = [2, 7, 11, 15]
    target = 9
    s = Solution()
    res = s.twoSum(nums, target)
    print(res)
