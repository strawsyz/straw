# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        new_head = ListNode()
        while head:
            new_head.val = head.val
            temp = ListNode()
            temp.next = new_head
            new_head = temp
        return new_head.next