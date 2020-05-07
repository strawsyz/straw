import numpy as np
from typing import List
# 虽然用了numpy而且效率不怎么样，但姑且能跑通

class Solution:
    def countSmaller(self, nums: List[int]) -> List[int]:
        if nums == []:
            return []
        nums = np.asarray(nums)

        results = []
        for index, num in enumerate(nums[:-1]):
            results.append(np.count_nonzero([nums[index + 1:] < num]))
        results.append(0)
        return results


if __name__ == '__main__':
    nums = [1,0,2]
    res = Solution().countSmaller(nums)
    print(res)
