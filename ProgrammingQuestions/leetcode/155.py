# 速度不怎么快但是可以跑通

class MinStack:
    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stack = []
        # 从大到小排列stack中的数字
        # 便于取最小值
        self.sorted = []

    def push(self, x: int) -> None:
        self.stack.append(x)
        if self.sorted == []:
            self.sorted.append(x)
        else:
            left, right = 0, len(self.sorted) - 1
            while left <= right:
                mid = left + (right - left) // 2
                if self.sorted[mid] > x:
                    left = mid + 1
                elif self.sorted[mid] < x:
                    right = mid - 1
                else:
                    left = mid  # 方便后面的处理
                    break
            last = self.sorted[-1]
            if left < len(self.sorted):
                for i in range(len(self.sorted) - 1, left, -1):
                    self.sorted[i] = self.sorted[i - 1]
                self.sorted[left] = x
                self.sorted.append(last)
            else:
                self.sorted.append(x)

    def pop(self) -> None:
        x = self.stack.pop()

        left, right = 0, len(self.sorted)
        while left <= right:
            mid = left + (right - left) // 2
            if self.sorted[mid] > x:
                left = mid + 1
            elif self.sorted[mid] < x:
                right = mid - 1
            else:
                left = mid
                break
        # 去掉left 上的那个值
        self.sorted = self.sorted[:left] + self.sorted[left + 1:]

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.sorted[-1]


# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(-2)
# obj.push(0)
# obj.push(-3)
# param_4 = obj.getMin()
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()
