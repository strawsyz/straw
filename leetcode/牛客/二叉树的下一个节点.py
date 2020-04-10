# -*- coding:utf-8 -*-
# class TreeLinkNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
#         self.next = None


class Solution:
    def GetNext(self, pNode):
        if not pNode:
            return None
        if pNode.right:
            # 如果有右子树，下一个节点就是右子树最左边的点
            pNode = pNode.right
            while pNode.left:
                pNode = pNode.left
            return pNode
        else:
            # 如果没有右子树
            while pNode.next:
                # 回到上一个父节点
                proot = pNode.next
                # 如果父节点的左子节点是就自己，说明自己在左子树
                if proot.left == pNode:
                    # 下一个节点就是父节点
                    return proot
                # 如果自己在右子树，说明自己所在的子树已经遍历完了
                # 回到父节点去
                pNode = pNode.next