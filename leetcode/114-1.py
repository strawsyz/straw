# Given a binary tree, flatten it to a linked list in-place.

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def flatten(self, root: TreeNode) -> None:
        """
        modify root in-place instead.
        """
        # 如果根节点是空。用于处理依赖出入的root就是None情况
        if root is None:
            # 就返回空
            return root
        left = right = None
        if root.left is not None:
            # 将左子树压平然后返回
            left = self.flatten(root.left)
        if root.right is not None:
            # 将右子树压平，然后返回
            right = self.flatten(root.right)
        if left:
            # 如果有左节点
            # 将左节点放到右节点的位置上
            root.right = left
            # 如果有右子树，就把右子树放到左子树的最右边
            if right:
                # 移动到左子树最右边的节点上
                while left.right:
                    left = left.right
                left.right = right
        elif right:
                # 如果只有右边，就把被压平的右子树放到右节点
                root.right = right
        # 将根节点的左子树设为空
        root.left = None
        return root
