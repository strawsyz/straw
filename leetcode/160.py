# Write a program to find the node at which the intersection of two singly linked lists begins.


# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        # 将两个链表按不同顺序拼接起来
        p = headA
        q = headB
        while p != q:
            p = p.next if p else headB
            q = q.next if q else headA
        return q


class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:

        p = headA
        hashmap = {}
        while p:
            hashmap[p] = 0
            p = p.next
        q = headB
        while q:
            res = hashmap.get(q)
            if res is not None:
                return q
            q = q.next