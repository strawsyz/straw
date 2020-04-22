# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

# 经过优化效率有所提升
# 和之前不同，直接遍历所有
class Solution:
    def isBalanced(self, root: TreeNode) -> bool:
        def main(root):
            if not root:
                return True, 0
            else:
                res_left, depth_left = main((root.left))
                res_right, depth_right = main((root.right))
                if abs(depth_left - depth_right) > 1:
                    return False, max(depth_right, depth_left) + 1
                else:
                    return True and res_left and res_right, max(depth_right, depth_left) + 1
        # result =
        return main(root)[0]


if __name__ == '__main__':
    s = Solution()
    root = TreeNode(1)
    root.left = TreeNode(1)
    root.left.left = TreeNode(1)
    res = s.isBalanced(root)
    print(res)