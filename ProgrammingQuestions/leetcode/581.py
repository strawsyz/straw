from typing import List

# 解法
# 先排序
# 然后比较排序前与排序后的不同
# 找到一个不同的位置和最后一个不同的位置
# 减一下得到答案

class Solution:
    def findUnsortedSubarray(self, nums: List[int]) -> int:
        length = len(nums)
        from copy import deepcopy
        arr = deepcopy(nums)
        nums.sort()
        # 初始如下，使得左右指针没有移动的情况下，最终结果返回0
        right = - 1
        left = 0
        # 找到第一个不同的数字
        for i in range(length):
            if arr[i] != nums[i]:
                left = i
                break
        # 找到最后一个不同的数字
        for i in range(length - 1, -1, -1):
            if arr[i] != nums[i]:
                right = i
                break
        return right - left + 1


if __name__ == '__main__':
    s = Solution()
    res = s.findUnsortedSubarray([])
    print(res)
