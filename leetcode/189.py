class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        import copy
        tmp = copy.deepcopy(nums)
        length = len(nums)
        k %= length
        for i in range(length):
            nums[i] = tmp[(i + length - k) % length]
