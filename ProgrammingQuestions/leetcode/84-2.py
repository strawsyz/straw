from typing import List


# 参考大佬的代码
# 栈的结构用的非常巧妙
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        # 最后加一个空的柱子
        heights.append(0)
        stack = [-1]
        largest = 0
        for i in range(len(heights)):
            # 如果第一个柱子要比最右边的柱子低
            while heights[i] < heights[stack[-1]]:
                # 弹出对应的柱子的高度
                h = heights[stack.pop()]
                w = i - stack[-1] - 1
                largest = max(largest, h * w)
            # 把新的柱子的下标存储进去
            stack.append(i)

        return largest


if __name__ == '__main__':
    res = Solution().largestRectangleArea([3, 423, 5, 222222222])
    print(res)
