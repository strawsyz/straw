# -*- coding:utf-8 -*-
class Solution:
    # 这里要特别注意~找到任意重复的一个值并赋值到duplication[0]
    # 函数返回True/False
    def duplicate(self, numbers, duplication):
        i = 0
        while i < len(numbers):
            # 先判断下标是否等于数字
            if numbers[i] == i:
                i += 1
                continue
            else:
                if numbers[numbers[i]] == numbers[i]:
                    # 因为要检查重复的数字，所以要给duplication赋值
                    duplication[0] = numbers[i]
                    return True
                temp = numbers[i]
                numbers[i], numbers[temp] = numbers[temp], numbers[i]
        return False

