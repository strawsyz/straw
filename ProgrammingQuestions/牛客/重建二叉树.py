# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

# 二叉树有四种遍历方法
# 先序遍历，中序遍历，后序遍历和层级遍历
# 根据中序遍历和其他任意一种遍历方法都可以重建唯一的二叉树
# 因为中序遍历可以根据根节点，将左右子树分开


class Solution:
    # 返回构造的TreeNode根节点
    def reConstructBinaryTree(self, pre, tin):
        # write code here
        root = self.build_tree(pre, tin)
        return root

    def build_tree(self, pre, tin):
        if not pre:
            return None
        # 前序遍历的第一个节点就是根节点
        x = pre[0]
        # 根据根节点的值将中序遍历的序列分为两个序列
        # 左子序列和右子序列
        p = tin.index(x)
        root = TreeNode(x)
        # 对左子树序列和右子树序列分别重建
        root.left = self.build_tree(pre[1:p + 1], tin[:p])
        root.right = self.build_tree(pre[p + 1:], tin[p + 1:])
        return root
