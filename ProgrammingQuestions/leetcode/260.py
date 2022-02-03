# Given an array of numbers nums, in which exactly two elements appear only once and
# all the other elements appear exactly twice.
# Find the two elements that appear only once.


class Solution:
    def singleNumber(self, nums: List[int]) -> List[int]:
        dict = {}
        res = []
        for num in nums:
            if num in dict:
                dict.pop(num)
                # res.append(num)
            else:
                dict[num] = 1
        for i in dict.keys():
            res.append(i)
        return res