import bisect

# 参考大佬的代码
# 解法：我们只要把输入数组反过来插入一个有序数组（降序）中，
# 插入的位置就是在原数组中位于它右侧的元素的个数。

class Solution:
    def countSmaller(self, nums: List[int]) -> List[int]:
        sortns = []
        res = []
        for n in reversed(nums):
            # 二分法插入
            idx = bisect.bisect_left(sortns, n)
            # 插入的位置就是在n右边且小于n的个数
            res.append(idx)
            sortns.insert(idx, n)
        return res[::-1]
