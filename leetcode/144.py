# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

# 使用栈结构来先序遍历二叉树
class Solution:
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        if not root:
            return []
        stack = [root]
        result = []
        while stack:
            node = stack.pop()
            result.append(node.val)
            # 因为栈是先进后出，所以要先放入右节点，再放入左节点
            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)
        return result
