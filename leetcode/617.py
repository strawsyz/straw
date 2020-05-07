# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

class Solution:
    def mergeTrees(self, t1: TreeNode, t2: TreeNode) -> TreeNode:
        if not t1 and not t2:
            # 如果两边为空
            return None
        result = 0
        if t1:
            # 将不为空的值加起来
            result += t1.val
        if t2:
            result += t2.val
        if not t1:
            # 如果节点为空，要创建新的节点，为了后续的运算
            t1 = TreeNode()
        if not t2:
            t2 = TreeNode()
        t1.val = result
        # 将整合好的子树插到原来的树上

        t1.right = self.mergeTrees(t1.right, t2.right)
        t1.left = self.mergeTrees(t1.left, t2.left)

        return t1
