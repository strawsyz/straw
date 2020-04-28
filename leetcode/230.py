# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution:
    def kthSmallest(self, root: TreeNode, k: int) -> int:
        def calcu_nodes(root):
            # 计算root为根节点的子树的所有节点数
            if root is None:
                return 0
            left = calcu_nodes(root.left)
            right = calcu_nodes(root.right)

            return left + right + 1

        while root:
            # 左子树的所有节点的数量加起来再加1等于当前节点的排名
            cur_rank = calcu_nodes(root.left) + 1
            if cur_rank == k:
                # 如果等于目标排名就直接返回
                return root.val
            elif cur_rank > k:
                # 如果当前节点排名大于目标值
                # 就跳到左子树
                root = root.left
            else:
                # 如果当前节点排序小于目标值
                # 就跳到右子树
                root = root.right
                # 重新设置目标值k
                # 因为跳到右子树的时候
                # 保证了里面的任何一个节点都至少大于cur_rank个数字
                k = k - cur_rank


if __name__ == '__main__':
    root = TreeNode(3)
    root.right = TreeNode(4)
    root.left = TreeNode(1)
    root.left.right = TreeNode(2)
    s = Solution()
    res = s.kthSmallest(root, 1)
    print(res)
