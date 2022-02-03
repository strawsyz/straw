# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

# 画个二叉树就能理解
# 一个节点左右节点的值相同
# 左节点的左节点要与右节点的右节点相同
# 左节点的右节点要与右节点的左节点相同
class Solution:
    def isSymmetrical(self, pRoot):
        # write code here
        if not pRoot:
            return True
        return self.is_symmertrical(pRoot.left, pRoot.right)

    def is_symmertrical(self, left, right):
        if not left and not right:
            # 如果两边都是空，说明两边相等
            return True
        if not left or not right:
            return False
        if left.val == right.val:
            return True & self.is_symmertrical(left.left, right.right) & self.is_symmertrical(left.right, right.left)
        else:
            return False
