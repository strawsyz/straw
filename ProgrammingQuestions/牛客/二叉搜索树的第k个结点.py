# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    # 返回对应节点TreeNode
    def KthNode(self, pRoot, k):
        # write code here
        stack = []

        # 先走到最左边的结点
        while pRoot:
            stack.append(pRoot)
            pRoot = pRoot.left

        while stack:
            node = stack.pop()
            k -= 1
            if k == 0:
                return node
            # 如果当前结点有右子树
            if node.right:
                node = node.right
                # 一直循环直到右子树的最右边
                while node:
                    stack.append(node)
                    node = node.left
        # 如果不返回None不知道为什么过不了所有的测试用例
        return None
