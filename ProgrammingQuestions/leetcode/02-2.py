# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

# 逻辑没有变化，就是看起来简洁了一点
class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        is_over = 0
        res = None
        while l1 or l2:
            if not l1:
                l1 = ListNode(0)
            if not l2:
                l2 = ListNode(0)
            temp = l1.val + l2.val + is_over
            if res:
                res.next = ListNode(temp % 10)
                res = res.next
            else:
                res = ListNode(temp % 10)
                result = res
            is_over = temp // 10
            l1, l2 = l1.next, l2.next
        if is_over:
            # 如果还有溢出的数字
            res.next = ListNode(1)
        return result
