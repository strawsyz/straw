# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def deleteDuplicates(self, head):
        new_head = ListNode(0)
        new_head.next = head
        # 由于最后要返回new_head,所以用temp引用new_head来计算
        temp = new_head
        while temp.next is not None and temp.next.next is not None:
            if temp.next.val == temp.next.next.val:
                # 如果遇到一个重复的数字
                temp_p, temp_v = temp.next, temp.next.val
                temp_p = temp_p.next
                while temp_p and temp_p.val == temp_v:
                    # 用循环把连续的所有重复的数字都处理了
                    temp_p = temp_p.next
                temp.next = temp_p
            else:
                # 如果没用重复就继续向后移动
                # 如果没用重复就继续向后移动
                temp = temp.next
        return new_head.next

