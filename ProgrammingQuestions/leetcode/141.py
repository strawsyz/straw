# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
# 快慢指针
class Solution:
    def hasCycle(self, head: ListNode) -> bool:
        slow, fast = head, head
        while fast:
            fast = fast.next
            if fast is None:
                return False
            else:
                fast = fast.next
                slow = slow.next
                if fast == slow:
                    return True
        return False
