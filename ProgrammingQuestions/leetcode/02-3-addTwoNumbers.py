# 思路遍历第一个链表，遍历完之后，第二个链表还有剩余的话，就继续遍历第二个链表
# 不如方法二好看

class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        idx = 0
        res = None  # tmp_result
        result = None
        plus_flag = 0
        while l1 is not None:
            if l2 is None:
                l2_val = 0
            else:
                l2_val = l2.val

            value = l1.val + l2_val + plus_flag
            plus_flag = value // 10
            node_value = value % 10

            if res is None:
                res = ListNode(val=node_value, next=None)
                result = res
            else:
                res.next = ListNode(val=node_value, next=None)
                res = res.next
            l1 = l1.next
            if l2 is not None:
                l2 = l2.next
            idx += 1

        while l2 is not None:

            if l2.val + plus_flag > 9:
                value = l2.val - 10 + plus_flag
                plus_flag = 1
            else:
                value = l2.val + plus_flag
                plus_flag = 0

            if res is None:
                res = ListNode(val=value, next=None)
                result = res
            else:
                res.next = ListNode(val=value, next=None)
                res = res.next
            l2 = l2.next

        if plus_flag == 1:
            res.next = ListNode(val=1, next=None)
        return result
