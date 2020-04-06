# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
from collections import deque
# 速度更快，但要使用第三方包

class Solution:
    def findBottomLeftValue(self, root: TreeNode) -> int:
        q = deque([root])
        while q:
            for i in range(len(q)):
                cur = q.popleft()
                if i == 0:
                    left_most = cur.val
                if cur.left:
                    q.append(cur.left)
                if cur.right:
                    q.append(cur.right)
        return left_most
