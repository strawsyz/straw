# 参考大佬的代码
class Solution:
    def trap(self, heights: List[int]) -> int:
        # left是左边最高的高度
        # right是右边最高的高度
        left_max = right_max = water = 0
        left, right = 0, len(heights) - 1
        while left <= right:
            left_max, right_max = max(left_max, heights[left]), max(right_max, heights[right])
            while left <= right and heights[left] <= left_max <= right_max:
                water += left_max - heights[left]
                left += 1
            while left <= right and heights[right] <= right_max <= left_max:
                water += right_max - heights[right]
                right -= 1
        return water
