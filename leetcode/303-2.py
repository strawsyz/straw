# 思路和之前的一样复杂度也一样
# 就是看起来简洁不少

class NumArray:
    def __init__(self, nums: List[int]):
        if not nums:
            self.results = []
            return
        pre = 0
        self.results = [pre]
        for i in range(len(nums)):
            pre += nums[i]
            self.results.append(pre)

    def sumRange(self, i: int, j: int) -> int:
        return self.results[j + 1] - self.results[i]
