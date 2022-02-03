# 使用冒泡排序，超级慢
class Solution():
    def findKthLargest(self, nums: List[int], k: int) -> int:
        length = len(nums)
        for i in range(k):
            for index in range(length - i - 1):
                if nums[index] > nums[index + 1]:
                    nums[index], nums[index + 1] = nums[index + 1], nums[index]

        return nums[-k]
