# -*- coding:utf-8 -*-
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution:
    def Serialize(self, root):
        result = []
        self.preorder(root, result)
        return '!'.join(result)

    def preorder(self, root, result):
        if not root:
            result.append('#')
        else:
            result.append(str(root.val))
            self.preorder(root.left, result)
            self.preorder(root.right, result)

    def Deserialize(self, s):
        nodes = s.split('!')
        root = self.build_tree(nodes)
        return root

    def build_tree(self, nodes):
        x = nodes.pop(0)
        if x == '#':
            return None
        else:
            root = TreeNode(int(x))
            root.left = self.build_tree(nodes)
            root.right = self.build_tree(nodes)
            return root

s =Solution()
res = s.Serialize(TreeNode(1))
print(res)
res = s.Deserialize(res)