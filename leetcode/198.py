class Solution:
    def rob(self, nums: List[int]) -> int:
        length = len(nums)
        dp = [0 for _ in range(length)]
        if length == 0:
            return 0
        if length == 1:
            return nums[0]
        dp[0] = nums[0]
        dp[1] = nums[1] if nums[1] > nums[0] else nums[0]
        if length == 2:
            return dp[1]
        # dp[2] = nums[1] if nums[1]>nums[0] + nums[2] else nums[0] + nums[2]
        for i in range(2, length):
            if dp[i - 1] > dp[i - 2] + nums[i]:
                dp[i] = dp[i - 1]
            else:
                dp[i] = dp[i - 2] + nums[i]
        return dp[length - 1]
