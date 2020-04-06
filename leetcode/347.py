from typing import List


# Given a non-empty array of integers, return the k most frequent elements.

class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        if len(nums) == 1:
            return nums
        if len(nums) == k:
            return nums
        hashmap = {}
        for cur_elem in nums:
            if cur_elem in hashmap:
                hashmap[cur_elem] += 1
            else:
                hashmap[cur_elem] = 1
        elems = []
        freqs_elems = []
        min_freq = 0
        # min_freq_elem = None
        for cur_elem in hashmap.keys():
            cur_freq = hashmap[cur_elem]
            if len(elems) == k:
                if cur_freq > min_freq:
                    # min_index = freqs_elems.index(min_freq)
                    min_freq_elem = elems[freqs_elems.index(min_freq)]

                    freqs_elems.remove(min_freq)
                    elems.remove(min_freq_elem)
                    freqs_elems.append(cur_freq)
                    min_freq = min(freqs_elems)
                    elems.append(cur_elem)
            else:
                # 还没有凑满k个元素
                if min_freq == 0:
                    # 如果还没有更新过最小频率数
                    min_freq = cur_freq
                    # min_freq_elem = cur_elem
                    # elems.append(i)
                    # freqs_elems.append(cur_freq)
                elif cur_freq < min_freq:
                    # 如果当前频率数要更加小，就更新频率数
                    # min_freq_elem = cur_elem
                    min_freq = cur_freq
                elems.append(cur_elem)
                freqs_elems.append(cur_freq)

        return elems


if __name__ == '__main__':
    te = [4, 1, -1, 2, -1, 2, 3]
    k = 2
    s = Solution()
    res = s.topKFrequent(te, k)
    print(res)
