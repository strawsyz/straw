# 参考官网解法，使用dp
# 会比之前的算法快很多
class Solution(object):
    def isMatch(self, text, pattern):
        # memo[i,j]存储text[i:]和pattern[j:]是否匹配
        memo = {}

        def dp(i, j):
            """计算text[i:]和pattern[j:]是否匹配"""
            if (i, j) not in memo:
                if j == len(pattern):
                    # 如果模式已经匹配完毕了
                    # 此时text还有剩余的话，就匹配失败
                    # 否则就匹配成功
                    ans = i == len(text)
                else:
                    # 文本还没有结束，并且模式的字符等于.或者等于文本的字符
                    # 则第一个字符匹配
                    first_match = i < len(text) and pattern[j] in {text[i], '.'}
                    # 模式还没有结束并且模式的下一个字符*
                    if j + 1 < len(pattern) and pattern[j + 1] == '*':
                        # dp(i, j+2)是*代表了0个字符的情况
                        # dp(i+1, j)代表了*匹配完了一个字符，还可以继续匹配，知道不再有相同的字符了
                        # 但前提是已经匹配了一个字符了
                        ans = dp(i, j + 2) or (first_match and dp(i + 1, j))
                    else:
                        ans = first_match and dp(i + 1, j + 1)
                # 当前的结果保存下来供下次使用
                memo[i, j] = ans
            return memo[i, j]

        return dp(0, 0)


if __name__ == '__main__':
    s = 'aahsiudhcfiwnaefnxzi'
    p = 'a*hosidefcoewjfiop'
    res = Solution().isMatch(s, p)
    print(res)
