from typing import List


# 使用了DFS来遍历所有的可能性
# 选决定第一个结果，然后获得第二个结果

class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        # 特殊情况处理
        if digits == '':
            return []
        # map = {2: ['a', 'b', 'c'],
        #        3: ['d', 'e', 'f'],
        #        4: ['g', 'h', 'i'],
        #        5: ['j', 'k', 'l'],
        #        6: ['m', 'n', 'o', ],
        #        7: ['p', 'q', 'r', 's'],
        #        8: ['t', 'u', 'v'],
        #        9: ['w', 'x', 'y', 'z']}
        # 下面的这种存储方式好像要快一点
        map = {2: 'abc',
               3: 'def',
               4: 'ghi',
               5: 'jkl',
               6: 'mno',
               7: 'pqrs',
               8: 'tuv',
               9: 'wxyz'}
        result = []

        # 深度查找
        def dfs(prefix, index):
            # 如果当前的d已经到达了输入字符串的长度，说明已经没有数字了
            if index == len(digits):
                # 将当前结果放入最后的结果中
                result.append(prefix)
            else:
                # 循环当前数组所对应的所有字符
                for x in map[int(digits[index])]:
                    # 加上前面留下的前缀和当前循环选择的字符，并将下标向后移动一位
                    dfs(prefix + x, index + 1)

        dfs('', 0)
        return result


if __name__ == '__main__':
    s = Solution()
    res = s.letterCombinations('')
    print(res)
