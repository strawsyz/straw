# 参考官方做法

class Solution:
    def firstMissingPositive(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """

        length = len(nums)

        if 1 not in nums:
            return 1
        elif length == 1:
            # 如果长度是1而且有数字1，说明一定是[2]
            return 2

        # 将小于1的整数和大于length的数字设置为1
        for i in range(length):
            if nums[i] <= 0 or nums[i] > length:
                nums[i] = 1

        # 使用索引和数字符号作为检查器
        for i in range(length):
            # 如果发现了一个数字 a - 改变第 a 个元素的符号
            a = abs(nums[i])
            if a == length:
                # 由于没有下标 n ，使用下标 0 的元素保存是否存在数字 n。
                nums[0] = - abs(nums[0])
            else:
                # 当读到数字 a 时，替换第 a 个元素的符号。
                nums[a] = - abs(nums[a])

        # 第一个正数的下标就是第一个缺失的数
        for i in range(1, length):
            if nums[i] > 0:
                return i

        if nums[0] > 0:
            # 由于使用了第一个位置来表示是否第length个数字
            return length
        # 如果全都没有确实就返回length+1
        return length + 1
