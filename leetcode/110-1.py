# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
# Given a binary tree, determine if it is height-balanced.
#
# For this problem, a height-balanced binary tree is defined as:
#
# a binary tree in which the left and right subtrees of every node differ in height by no more than 1.

class Solution:
    def isBalanced(self, root: TreeNode) -> bool:
        if root is None:
            return True
        left_depth = self.depth(root.left)
        right_depth = self.depth(root.right)
        if abs(left_depth - right_depth) > 1:
            return False
        else:
            return self.isBalanced(root.left) and self.isBalanced(root.right)

    # 测量树的深度
    def depth(self, root):
        if root is None:
            return 0

        left_depth = self.depth(root.left)
        right_depth = self.depth(root.right)

        return max(left_depth, right_depth) + 1

