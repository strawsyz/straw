# 删除链表中重复的节点
# 没有作对
class Solution:
    def deleteDuplication(self, pHead):
        # write code here
        last = pHead
        now = pHead.next
        flag = False
        head_flag = False
        while last.val == now.val and now is not None:
            # pHead = now
            head_flag = True
            # last = now
            # temp = pHead
            # last = last.next
            now = now.next
            last.next = now
        if now is None:
            return None
        # if flag:
        #     flag = False
        #     pHead = now
        #     last = last.next
        #     now = now.next
        last_last = pHead
        while now is not None:
            if last.val == now.val:
                flag = True
                # last.next = now.next
                temp = last
                now = now.next
            else:
                if flag == True:
                    flag = False
                    now = now.next
                    # last.next = now
                    last_last.next = now
                    # last_last.next = now
                    # last = last_last
                    # last_last.next = last
                else:
                    last = last.next
                    last_last = last_last.next
                    now = now.next
                    # last_last = last_last.next

                    # now = now.next
        if head_flag:
            pHead = pHead.next
        return pHead
