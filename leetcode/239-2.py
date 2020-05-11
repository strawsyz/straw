from collections import deque


# 参考大佬的代码
class Solution:
    def maxSlidingWindow(self, nums, k):
        res = []
        # 存储index，从小到大排序，将nums中的索引放入队列
        # 第一个索引，记录当前窗口最大的数字的索引
        indices = deque()
        for i, num in enumerate(nums):
            # 遍历所有的数字
            while indices and nums[indices[-1]] <= num:
                # 从记录下的索引开始，从右到做判断
                # 如果索引对应的数字小于或等于当前的数字，就扔掉
                # 从而确保最右边的索引对应的数字是当前滑动窗口中最小的数字
                indices.pop()
            # 加新的坐标添加到队列中
            indices += [i]
            # 确保最左边的数值还没有超过k
            if i - indices[0] >= k:
                # 将第一个数字扔掉
                indices.popleft()
            if i + 1 >= k:
                # 如果满足滑动窗口的尺寸
                res.append(nums[indices[0]])
        return res
