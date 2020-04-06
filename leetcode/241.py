from typing import List


class Solution:
    def diffWaysToCompute(self, input: str) -> List[int]:
        res = []
        length = len(input)
        for i in range(length):
            c = input[i]
            if c in ['+', '-', '*']:
                for l in self.diffWaysToCompute(input[:i]):
                    for r in self.diffWaysToCompute(input[i + 1:]):
                        if c == '+':
                            res.append(l + r)
                        elif c == '-':
                            res.append(l - r)
                        else:
                            res.append(l * r)
        if res == []:
            return [int(input)]

        return res

#
class Solution:
    def diffWaysToCompute(self, input: str) -> List[int]:
        signal = {
            '+': lambda x, y: x + y,
            '-': lambda x, y: x - y,
            '*': lambda x, y: x * y
        }

        def compute(input):
            res = []
            n = len(input)
            for i in range(n):
                if input[i] in signal:
                    for left in compute(input[:i]):
                        for right in compute(input[i + 1:]):
                            res.append(signal[input[i]](left, right))
            if res == []:
                return [int(input)]
            return res

        return compute(input)


if __name__ == '__main__':
    s = Solution()
    inp = '2-1-1'
    res = s.diffWaysToCompute(inp)
    print(res)
