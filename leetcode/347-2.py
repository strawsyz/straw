from typing import List


# Given a non-empty array of integers, return the k most frequent elements.
# 使用了桶排序，和不使用桶排序没有太大区别

class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        freq_elems = {}
        for num in nums:
            freq_elems[num] = freq_elems.get(num, 0) + 1
        buckets = [[] for i in range(len(nums) + 1)]
        # buckets的索引表示频率数
        for key, value in freq_elems.items():
            buckets[value].append(key)

        result = []

        for x in range(len(nums), -1, -1):
            if k > 0 and buckets[x]:
                result += buckets[x]
                k -= len(buckets[x])
            if k == 0:
                return result


if __name__ == '__main__':
    te = [4, 1, -1, 2, -1, 2, 3]
    k = 2
    s = Solution()
    res = s.topKFrequent(te, k)
    print(res)
