# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

# 使用递归来实现
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        # 用于存储新链表的头部
        self.head
        def recur(head):
            if head:
                # 如果本节点不为空，创建新的节点
                new_node = ListNode(head.val)
                # 递归
                res = recur(head.next)
                if res:
                    # 如果递归结果不为空，说明下一个节点不为空
                    res.next = new_node
                    return res.next
                else:
                    # 递归返回空。说明当前节点是最后一个节点
                    self.head = new_node
                    return new_node
            else:
                # 如果节点为空，就返回空
                return

        recur(head)
        return self.head


if __name__ == '__main__':
    head = NodeList(1)
    head.next = NodeList(2)
    head.next.next = NodeList(3)
    head.next.next.next = NodeList(4)
    Solution.reverseList(head)