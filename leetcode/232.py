# Implement the following operations of a queue using stacks.
#
# push(x) -- Push element x to the back of queue.
# pop() -- Removes the element from in front of queue.
# peek() -- Get the front element.
# empty() -- Return whether the queue is empty.

# 其实做错了，因为没有使用栈来实现队列
class MyQueue:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.nums = []

    def push(self, x: int) -> None:
        """
        Push element x to the back of queue.
        """
        self.nums.append(x)

    def pop(self) -> int:
        """
        Removes the element from in front of queue and returns that element.
        """
        # temp = nums[0]
        # self.nums.remove(temp)
        if len(self.nums) == 0:
            return None
        return self.nums.pop(0)

    def peek(self) -> int:
        """
        Get the front element.
        """
        if len(self.nums) == 0:
            return None
        return self.nums[0]

    def empty(self) -> bool:
        """
        Returns whether the queue is empty.
        """
        return len(self.nums) == 0
# Your MyQueue object will be instantiated and called as such:
# obj = MyQueue()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.peek()
# param_4 = obj.empty()
