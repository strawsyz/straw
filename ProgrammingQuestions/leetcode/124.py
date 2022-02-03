# Given a non-empty binary tree, find the maximum path sum.
#
# For this problem, a path is defined as any sequence of nodes from
#  some starting node to any node in the tree along the parent-child
# connections. The path must contain at least one node and does not
# need to go through the root.

# 从任何连着的几个点就行，不需要是从某个点走到另一个点的没有重复经过某个任意点的直线
# 一开始因为理解错了，做错了

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

# 引用了大佬的代码
# https://leetcode.com/problems/binary-tree-maximum-path-sum/discuss/602827/Python-recursive-clean-beat-99

class Solution(object):
    def maxPathSum(self, root):
        # 初始化为无穷小
        self.max = float('-inf')

        def get_sum(root):
            if not root:
                # 如果为空。空节点的和是0
                return 0
            else:
                # 如果是负数就不添加
                ls = max(get_sum(root.left), 0)
                # 如果是负数就不添加
                rs = max(get_sum(root.right), 0)
                # 如果大于目前为止的最大值，就更新最大值
                self.max = max(self.max, ls + rs + root.val)
                # 返回经过root节点，以root节点为根的子树的节点
                # 连接起来所能达到的最大值
                return max(ls, rs, 0) + root.val

        get_sum(root)
        return self.max