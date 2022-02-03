# 给定一个链表，返回链表开始入环的第一个节点。 如果链表无环，则返回 null。
# 为了表示给定链表中的环，我们使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。
# 如果 pos 是 -1，则在该链表中没有环。
# 说明：不允许修改给定的链表。

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

# 使用快慢指针

class Solution:
    def detectCycle(self, head: ListNode) -> ListNode:
        fast, slow = head, head
        while fast is not None:
            # 快指针
            fast = fast.next
            # 如果走到底了就跳出
            if fast is None:
                break
            # 快指针走两步
            fast = fast.next
            # 慢指针走一步
            slow = slow.next
            # 重叠的时候就跳出去
            if slow == fast:
                break
        # 使用遇到了空值而跳出循环的话
        if fast is None:
            # 说明这个链表没有形成环
            return None
        # 如果slow和另一个指针从头出发。一步一步的走
        # 直到相遇，相遇的位置就出现环的位置
        while slow != head:
            slow = slow.next
            head = head.next
        return slow
