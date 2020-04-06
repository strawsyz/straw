# Given a binary tree, flatten it to a linked list in-place.

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def flatten(self, root: TreeNode) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        while root:
            if not root.left:
                # 如果没有左节点
                # 就遍历右节点
                root = root.right
            else:
                pre = root.left
                while pre.right:
                    pre = pre.right
                #  将右子树放到左子树的最右边上
                pre.right = root.right
                #  将左子树放到右子树上
                root.right = root.left
                # let left tree is None
                root.left = None
                root = root.right



class Solution:
    def flatten(self, root: TreeNode) -> None:
        """
        modify root in-place instead.
        """
        if root is None:
            return root
        left = right = None
        if root.left is not None:
            left = self.flatten(root.left)
        if root.right is not None:
            right = self.flatten(root.right)
        if left:
            root.right = left
            while left.right:
                left = left.right
            if right:
                left.right = right
        else:
            if right:
                root.right = right
        root.left = None
        return root
