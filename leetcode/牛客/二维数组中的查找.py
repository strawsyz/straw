# -*- coding:utf-8 -*-

# 时间复杂度0（m+n）
class Solution:
    # array 二维列表
    def Find(self, target, array):
        # write code here
        # 找到左下角的点
        left_row, left_col = len(array) - 1, 0
        # 找到右上角的点
        right_row, right_col = 0, len(array[0]) - 1
        while left_row >= right_row and left_col <= right_col:
            if array[left_row][left_col] < target:
                left_col += 1
            elif array[left_row][left_col] > target:
                left_row -= 1
            else:
                return True
            if array[right_row][right_col] < target:
                right_row += 1
            elif array[right_row][right_col] > target:
                right_col -= 1
            else:
                return True
        return False
