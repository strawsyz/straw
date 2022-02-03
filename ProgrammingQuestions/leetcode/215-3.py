from typing import List
import heapq


# 使用快速排序

class Solution():
    # 自己实现的快速排序
    def findKthLargest(self, nums: List[int], k: int) -> int:
        def partition(nums, left, right):
            # 返回nums[left]在nums中的排名
            # 按从大到小排序
            # 最左边的值作为哨兵
            pivot = nums[left]
            pivot_index = left
            # 左指针向右移动一步
            left += 1
            #  两个指针相遇，跳出循环
            while right >= left:
                if nums[right] <= pivot:
                    # 如果右指针要比哨兵指的数小
                    # 就继续向左移动
                    right -= 1
                    continue
                if nums[left] >= pivot:
                    # 如果左指针的数要比哨兵的大
                    # 就继续向右移动
                    left += 1
                    continue
                # 右边大于哨兵，且左边小于哨兵，就左右交换位置
                nums[right], nums[left] = nums[left], nums[right]
                left += 1
                right -= 1
            nums[right], nums[pivot_index] = nums[pivot_index], nums[right]
            # 右指针表示哨兵在数组第right大
            return right

        left = 0
        right = len(nums) - 1
        while True:
            idx = partition(nums, left, right)
            if idx == k - 1:
                return nums[idx]
            # 根据得到的idx，调整左右指针的位置
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
