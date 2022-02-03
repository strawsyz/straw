from typing import *


class Solution:
    # 这题很自然地想到归并排序，再取中间数，但是是nlogn的复杂度，题目要求logn
    # 所以要用二分法来巧妙地进一步降低时间复杂度
    # 思想就是利用总体中位数的性质和左右中位数之间的关系来把所有的数先分成两堆，然后再在两堆的边界返回答案
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        len_1 = len(nums1)
        len_2 = len(nums2)
        # 让nums2成为更长的那一个数组
        if len_1 > len_2:
            nums1, nums2, len_1, len_2 = nums2, nums1, len_2, len_1

        # 如果两个都为空的异常处理
        if len_2 == 0:
            return

        # nums1中index在imid左边的都被分到左堆，nums2中jmid左边的都被分到左堆
        left_1, right_1 = 0, len_1

        # 将所有数字分到左边和右边，左边的数字全都比右边的数字小
        while True:
            # mid_1左边的部分属于左边
            mid_1 = left_1 + (right_1 - left_1) // 2
            # 让左右堆大致相等需要满足的条件是mid_1 + mid_2 = len_1 + len_2 - mid_1 - mid_2
            # 即 mid_2 = (len_1 + len_2 - 2 * mid_1)//2
            mid_2 = (len_1 + len_2 - 2 * mid_1) // 2

            # 前面的判断条件只是为了保证不会index out of range
            if mid_1 > 0 and nums1[mid_1 - 1] > nums2[mid_2]:
                right_1 = mid_1 - 1
            elif mid_1 < len_1 and nums2[mid_2 - 1] > nums1[mid_1]:
                left_1 = mid_1 + 1
            # 满足条件
            else:
                # 右边最小的值
                if mid_1 == len_1:
                    # 如果nums1中的所有数字都被分到左边
                    min_right = nums2[mid_2]
                elif mid_2 == len_2:
                    # 如果nums2的所有数字都被分到左边
                    min_right = nums1[mid_1]
                else:
                    min_right = min(nums1[mid_1], nums2[mid_2])
                # 计算左边最大值
                if mid_1 == 0:
                    max_left = nums2[mid_2 - 1]
                elif mid_2 == 0:
                    max_left = nums1[mid_1 - 1]
                else:
                    max_left = max(nums1[mid_1 - 1], nums2[mid_2 - 1])

                # 判断两个数组的长度，
                if (len_1 + len_2) % 2 == 1:
                    return min_right
                else:
                    return (max_left + min_right) / 2


if __name__ == '__main__':
    nums2 = [1, 3]
    nums1 = [2]
    solution = Solution()
    output = solution.findMedianSortedArrays(nums1, nums2)
    print(output)
