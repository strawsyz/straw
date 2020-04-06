# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
# 巧妙！
class Solution:
    def findBottomLeftValue(self, root: TreeNode) -> int:
        q = [root]
        while q:
            root = q.pop(0)
            if root.right:
                q.append(root.right)
            if root.left:
                q.append(root.left)
        return root.val
