# Write a program to find the node at which the intersection of two singly linked lists begins.

# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


# 先把A用字典保存起来，然后用B中的数字一个个判断是否在字典中存在
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
