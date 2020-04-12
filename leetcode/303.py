class NumArray:
    def __init__(self, nums: List[int]):
        if not nums:
            self.results = []
            return
        self.results = [nums[0]]
        for i in range(1, len(nums)):
            self.results.append(self.results[-1] + nums[i])

    def sumRange(self, i: int, j: int) -> int:
        if i == 0:
            return self.results[j]
        else:
            return self.results[j] - self.results[i - 1]
            # Your NumArray object will be instantiated and called as such:
            # obj = NumArray(nums)
            # param_1 = obj.sumRange(i,j)
