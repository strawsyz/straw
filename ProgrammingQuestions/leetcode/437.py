# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def pathSum(self, root: TreeNode, sum: int) -> int:

        def dfs(root, cur):
            # 求解以root为节点，合等于sum的可能性
            if root:
                result = 0
                cur += root.val
                if cur == sum:
                    result += 1
                result += dfs(root.left, cur)
                result += dfs(root.right, cur)
                return result
            else:
                return 0

        def dfs_2(root):
            # 使用深度遍历，分别让树上的每个非空节点调用dfs函数，
            # 来就算以当前作为起点的和为sum的个数
            if root:
                result = 0
                result += dfs(root, 0)
                result += dfs_2(root.right)
                result += dfs_2(root.left)
            else:
                return 0
            return result

        return dfs_2(root)


if __name__ == '__main__':
    root = TreeNode(1)
    head = root
    root.right = TreeNode(2)
    root = root.right
    root.right = TreeNode(3)
    root = root.right
    root.right = TreeNode(4)
    root = root.right
    root.right = TreeNode(5)

    s = Solution()
    res = s.pathSum(head, 3)
    print(res)
