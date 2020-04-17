# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

# 逻辑和之前的一样，看起来简洁了一点

class Solution:
    def isBalanced(self, root: TreeNode) -> bool:
        def depth(root):
            if not root:
                return 0
            else:
                return max(depth(root.left), depth(root.right)) + 1

        if root is None:
            return True
        if abs(depth(root.left) - depth(root.right)) > 1:
            return False
        else:
            return self.isBalanced(root.left) and self.isBalanced(root.right)

