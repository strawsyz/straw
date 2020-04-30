# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

# 递归的方法

class Solution:
    def invertTree(self, root: TreeNode) -> TreeNode:
        if not root:
            return None
        else:
            # 先翻转左右子树
            right = self.invertTree(root.right)
            left = self.invertTree(root.left)
            # 将翻转后的左右子树交换位置
            root.left = right
            root.right = left

            return root
