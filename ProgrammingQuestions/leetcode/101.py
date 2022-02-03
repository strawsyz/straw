# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

# 使用深度遍历。判断二叉树是否对称

class Solution:
    def isSymmetric(self, root: TreeNode) -> bool:
        def dfs(left, right):
            if not left:
                # 如果左边为空
                if not right:
                    # 且右边为空，则相同
                    return True
                else:
                    # 但右边不为空，则不相同
                    return False
            if not right:
                # 如果左边部位不为空，但是右边是空，就返回空
                return False
            else:
                # 两边都不为空
                result = left.val == right.val
            if result:
                # 如果两边相同，则继续判断子节点
                result = dfs(left.left, right.right)
                if result:
                    return dfs(left.right, right.left)
                else:
                    return False
            else:
                # 如果左右不相同，直接返回False
                return False
        if root:
            return dfs(root.left, root.right)
        else:
            # 如果是空节点，直接返回正确
            return True
