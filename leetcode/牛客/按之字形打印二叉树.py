# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def Print(self, pRoot):
        # write code here

        i = 0
        q = [pRoot]
        result = []
        while q != []:
            root = q.pop()
            result.append(root.val)
            print(root.val)
            if i % 2 == 0:
                if root.right:
                    q.append(root.right)
                if root.left:
                    q.append(root.left)
            else:
                if root.left:
                    q.append(root.left)
                if root.right:
                    q.append(root.right)
            i+=1
        return result