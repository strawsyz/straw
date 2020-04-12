from typing import List


# 使用三指针

class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        left = mid = 0
        right = len(nums) - 1
        # 注意是小于等于
        while mid <= right:
            if nums[mid] == 0:
                nums[left], nums[mid] = nums[mid], nums[left]
                left += 1
                mid += 1
            elif nums[mid] == 1:
                mid += 1
            else:
                nums[right], nums[mid] = nums[mid], nums[right]
                right -= 1


if __name__ == '__main__':
    s = Solution()
    nums = [2, 0, 1]
    s.sortColors(nums)
    print(nums)
