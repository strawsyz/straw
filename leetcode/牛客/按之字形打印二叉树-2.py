# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def Print(self, pRoot):
        # write code here
        if not pRoot:
            return []

        i = 1  # 第几行
        q = [pRoot]
        stack = []  # 偶数行栈存储
        result = [[]]
        nowlast = pRoot  # 当前层最后一个节点
        nextlast = None  # 下一个层最后一个节点
        while q:
            node = q.pop(0)
            if i % 2 == 1:
                result[-1].append(node.val)
            else:
                stack.append(node.val)
            if node.left:
                nextlast = node.left
                q.append(node.left)
            if node.right:
                nextlast = node.right
                q.append(node.right)
            if node == nowlast:
                nowlast = nextlast
                if i % 2 == 0:
                    stack.reverse()
                    result.append(stack)
                    stack = []
                    result.append([])
                i += 1

        if result[-1]:
            return result
        else:
            return result[:-1]
