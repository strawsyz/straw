# 使用了快慢指针


class Solution:
    def isPalindrome(self, head):
        slow, fast = head, head
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
        half = []
        while slow:
            half += slow.val
            slow = slow.next
        half = half[::-1]
        for i in range(len(half)):
            if head.val == half[i]:
                head = head.next
            else:
                return False
        return True