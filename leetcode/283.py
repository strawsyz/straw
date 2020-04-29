from typing import List


class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        index = 0
        zero_num = 0
        for num in nums:
            if num == 0:
                zero_num += 1
            else:
                nums[index - zero_num] = num
            index += 1
        for i in range(index - zero_num, index):
            nums[i] = 0
        return nums


if __name__ == '__main__':
    res = Solution().moveZeroes([1])
    print(res)
