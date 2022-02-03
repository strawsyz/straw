from typing import List


# 没有使用dfs，采用循环的方法

class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        map = {2: 'abc', 3: 'def', 4: 'ghi', 5: "jkl", 6: 'mno', 7: 'pqrs', 8: 'tuv', 9: 'wxyz'}
        if digits == '':
            return []
        result = ['']

        def add_digit(digit, result):
            new_result = []
            for c in map[int(digit)]:
                for i in range(len(result)):
                    new_result.append(result[i] + c)
            return new_result

        for digit in digits:
            result = add_digit(digit, result)
        return result


if __name__ == '__main__':
    s = Solution()
    res = s.letterCombinations('32')
    print(res)
