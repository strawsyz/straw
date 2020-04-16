# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None


class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        res = ListNode()
        if
        while l1 or l2:
            if not l1:
                l1 = ListNode(0)
            if not l2:
                l2 = ListNode(0)
            temp = l1.val + l2.val
            res.val = temp//10

        is_plus_one = False
        res = None
        while l1 or l2:
            if l1 is None:
                l1 = ListNode(0)
                temp = l2.val
            elif l2 is None:
                l2 = ListNode(0)
                temp = l1.val
            else:
                temp = l1.val + l2.val
            if is_plus_one == True:
                temp = temp + 1 
            if temp > 9:
                temp = temp - 10
                is_plus_one = True
            else:
                is_plus_one = False
            if res:
                res.next = ListNode(temp)
                res = res.next
            else:
                res = ListNode(temp)
                result = res
            l1, l2 = l1.next, l2.next
        if is_plus_one == True:
            res.next = ListNode(1)
        return result
