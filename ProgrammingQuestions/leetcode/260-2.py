# Given an array of numbers nums, in which exactly two elements appear only once and
# all the other elements appear exactly twice.
# Find the two elements that appear only once.


class Solution:
    def singleNumber(self, nums: List[int]) -> List[int]:
        diff = 0
        for num in nums:
            diff = diff ^ num
        diff = diff & (-diff)
        ret = [0, 0]
        for num in nums:
            if diff & num == 0:
                ret[0] = ret[0] ^ num
            else:
                ret[1] = ret[1] ^ num
        return ret
