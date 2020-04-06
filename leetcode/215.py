from typing import List
import heapq


class Solution():

    def quicksort(self, nums, left, right):
        if left == right:
            return nums[left]
        pivot = self.partition(nums, left, right)
        if pivot == self._k:
            return nums[pivot]
        elif pivot < self._k:
            return self.quicksort(nums, pivot + 1, right)
        else:
            return self.quicksort(nums, left, pivot - 1)

    def partition(self, nums, left, right):
        pivot = nums[left]
        while left < right:
            while nums[right] >= pivot:
                right -= 1
            if left < right:
                nums[left] = nums[right]
                left += 1

    #    todo 未完成

    # def findKthLargest(self, nums: List[int], k: int) -> int:
    #     return sorted(nums, reverse=True)[k - 1]

    # def findKthLargest(self, nums: List[int], k: int) -> int:
    #     return heapq.nlargest(k, nums)[-1]

    # 自己实现的快速排序
    def findKthLargest(self, nums: List[int], k: int) -> int:
        def partition(nums, left, right):
            pivot = nums[left]
            pivot_index = left
            left += 1
            while right >= left:
                #                 两个指针相遇，跳出循环
                if nums[right] <= pivot:
                    right -= 1;
                    continue
                if nums[left] >= pivot:
                    left += 1;
                    continue
                nums[right], nums[left] = nums[left], nums[right]
                left += 1
                right -= 1
            nums[right], nums[pivot_index] = nums[pivot_index], nums[right]
            return right

        left = 0
        right = len(nums) - 1
        while True:
            idx = partition(nums, left, right)
            if idx == k - 1:
                return nums[idx]
            if idx < k - 1:
                left = idx + 1
            if idx > k - 1:
                right = idx - 1


if __name__ == '__main__':
    s = Solution()
    nums = [3, 2, 1, 5, 6, 4]
    k = 2
    res = s.findKthLargest(nums, k)
    print(res)
