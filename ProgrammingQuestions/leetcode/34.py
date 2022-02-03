from typing import List


class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:

        left = 0
        right = len(nums) - 1
        res_left = res_right = -1
        length = len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] < target:
                left = mid + 1
            elif nums[mid] > target:
                right = mid - 1
            else:
                res_left = res_right = mid

                while res_left != 0 and nums[res_left - 1] == target:
                    res_left = res_left - 1
                while res_right != length and nums[res_right + 1] == target:
                    res_right = res_right + 1
                break

        return [res_left, res_right]


if __name__ == '__main__':
    nums = [1]
    target = 1
    s = Solution()
    res = s.searchRange(nums, target)
    print(res)
