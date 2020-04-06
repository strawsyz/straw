# Given a n x n matrix where each of the rows and columns are sorted in ascending order,
# find the kth smallest element in the matrix.
#
# Note that it is the kth smallest element in the sorted order, not the kth distinct element.
import heapq
from typing import List

# 使用堆排序
class Solution:
    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
        n = len(matrix)
        h = []
        for i in range(n):
            heapq.heappush(h, (matrix[0][i], 0, i))
        for i in range(k - 1):
            item = heapq.heappop(h)
            if item[1] + 1 < n:
                heapq.heappush(h, (matrix[item[1] + 1][item[2]], item[1] + 1, item[2]))
        return heapq.heappop(h)[0]
