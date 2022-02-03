# Given a binary tree, flatten it to a linked list in-place.

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

# 代码更加简洁

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
                # 如果有左节点
                pre = root.left
                # 移动到左子树的最右边
                while pre.right:
                    pre = pre.right
                # 将右子树放到左子树的最右边上
                pre.right = root.right
                #  将左子树放到右子树上
                root.right = root.left
                # 让左子树为空
                root.left = None
                root = root.right
