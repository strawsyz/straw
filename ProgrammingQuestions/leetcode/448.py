from typing import List


# nums数组中范围在1到n的整数，n是数组长度！
class Solution:
    def findDisappearedNumbers(self, nums: List[int]) -> List[int]:
        for n in nums:
            while n > 0:
                tmp = nums[n-1]
                nums[n-1] = -1
                n = tmp
        result = []
        for i, num in enumerate(nums):
            # 数字大于0的位置都是元数组中不存在的数字
            if num > 0:
                result.append(i + 1)
        return result


if __name__ == '__main__':
    print(Solution().findDisappearedNumbers([4, 3, 2, 7, 8, 2, 3, 1]))
