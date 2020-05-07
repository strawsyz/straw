from typing import List


# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


# 比较暴力的方法，不会超时

class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        def merge(list1, list2):
            """合并两个链表"""
            if list1.val is None:
                return list2
            elif list2.val is None:
                return list1
            head = list1
            pre_list1 = None
            # 比较两个数组
            while list1 and list2:
                if list1.val < list2.val:
                    pre_list1 = list1
                    list1 = list1.next
                else:
                    if pre_list1 is None:
                        head = list2
                        list2 = list2.next
                        head.next = list1
                        pre_list1 = head
                    else:
                        temp = list2
                        list2 = list2.next
                        pre_list1.next = temp
                        temp.next = list1
                        pre_list1 = temp
            if list2:
                pre_list1.next = list2
            return head

        # 去掉列表中的空链表
        index_none = []
        for i, list_ in enumerate(lists):
            if list_ is None:
                index_none.append(i)
        temp = 0
        for index in index_none:
            index -= temp
            lists.pop(index)
            temp += 1

        if len(lists) == 0:
            return
        list1 = lists[0]
        for list_ in lists[1:]:
            list1 = merge(list1, list_)
        return list1


if __name__ == '__main__':
    input = []
    head = ListNode(1)
    head.next = ListNode(4)
    head.next.next = ListNode(5)
    input.append(head)
    head = ListNode(1)
    head.next = ListNode(3)
    head.next.next = ListNode(4)
    input.append(head)
    Solution().mergeKLists(input)
