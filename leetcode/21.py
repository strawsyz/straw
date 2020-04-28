# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

# 合并两个有序列表
# 要注意，一次比较只能向后移动一个节点

class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        head = ListNode(0)
        head_4_result = head
        while l1 and l2:
            if l1.val > l2.val:
                # 先把小的值赋给下一个值
                head.next = l2
                # head的指针向后移动一位
                head = head.next
                l2 = l2.next
            else:
                head.next = l1
                head = head.next
                l1 = l1.next

        if l1 and not l2:
            head.next = l1
        if l2 and not l1:
            head.next = l2
        # 如果两边都没有返回值了
        # 返回空头部节点下的下一个节点
        return head_4_result.next


if __name__ == '__main__':
    l1 = ListNode(1)
    l1.next = ListNode(2)
    l1.next.next = ListNode(3)
    l2 = ListNode(1)
    l2.next = ListNode(4)
    l2.next.next = ListNode(5)
    Solution().mergeTwoLists(l1, l2)
