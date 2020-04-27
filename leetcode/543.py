# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

# 由于叠加（递归）了两次深度遍历导致速度比较慢
# 还可以优化

class Solution:
    def diameterOfBinaryTree(self, root: TreeNode) -> int:
        def depth(root):
            # 测量深度
            if not root:
                # 如果节点为空，则长度是0
                return 0
            # 左右子树的最大深度加上1，等于root的深度
            return max(depth(root.left), depth(root.right)) + 1

        def length(root):
            # 计算root的直径
            if not root:
                # 如果节点为空。则直径是0
                return 0
            # 子树的深度等于子树最底层的叶节点走到顶部的步数
            # 从左子树的底层走到顶部的步数加上从顶部走到右子树最底层的步数
            # 等于root的直径
            return depth(root.left) + depth(root.right)

        def find_max_length(root):
            # 找到root的最长直径
            if root:
                # 当前节点为根和以左右字数分别为根的情况下的直径的最大值
                # 等于root的最大直径
                return max([length(root), find_max_length(root.left), find_max_length(root.right)])
            else:
                # 如果root为空，最大直径就是0
                return 0

        return find_max_length(root)