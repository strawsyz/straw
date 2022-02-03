# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

# 使用两个深度遍历实现
# 可以使用字符串代替数字来优化
class Codec:
    def serialize(self, root):
        """Encodes a tree to a single string.

        :type root: TreeNode
        :rtype: str
        """
        if root is None:
            return []
        queue = [root]
        data = []

        def dfs():
            if queue == []:
                return
            node = queue.pop(0)
            data.append(node.val)
            if node.left:
                queue.append(node.left)
                # 添加节点之后就先处理添加的节点
                dfs()
            else:
                data.append(None)
            if node.right:
                queue.append(node.right)
                dfs()
            else:
                data.append(None)

        dfs()
        return data

    def deserialize(self, data):
        """Decodes your encoded data to tree.

        :type data: str
        :rtype: TreeNode
        """

        def dfs():
            if data == []:
                return None
            tmp = data.pop(0)
            if tmp is None:
                return None
            else:
                node = TreeNode(tmp)
                node.left = dfs()
                node.right = dfs()
                return node

        return dfs()

        # Your Codec object will be instantiated and called as such:
        # codec = Codec()
        # codec.deserialize(codec.serialize(root))
