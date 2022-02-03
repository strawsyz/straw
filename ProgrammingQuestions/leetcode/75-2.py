from typing import List


# 使用三指针

class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        # left的左边都是0，mid到left之间都是1
        left = mid = 0
        # right的右边都是2
        right = len(nums) - 1
        # 注意是小于等于
        # 通过移动mid和right来遍历nums
        while mid <= right:
            if nums[mid] == 0:
                # 如果当前的mid指针指着0就把left和mid指针的值分别调换位置
                nums[left], nums[mid] = nums[mid], nums[left]
                # 左指针向右移动一步
                left += 1
                # 中间指针也向右移动一步
                mid += 1
            elif nums[mid] == 1:
                # 如果是1，只要中间指针向右移动一步就可以了
                mid += 1
            else:
                # 如果是2，需要将mid指着的数字和right的数字交换位置
                # 但是不能中间指针向右移动，因为有可能right指针不来就指着一个2
                nums[right], nums[mid] = nums[mid], nums[right]
                #右指针向左移动一位
                right -= 1


if __name__ == '__main__':
    s = Solution()
    nums = [2, 0, 1]
    s.sortColors(nums)
    print(nums)
