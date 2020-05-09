# 参考官网的解法

class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        if not p:
            # 如果没有模式
            # 输入的s也是空的话，就匹配
            # 否则就不匹配
            return not s
        # 每次只判断第一个字符是否匹配
        # 先不考虑星号
        # bool()只会把空字符串转为False
        if bool(s) and p[0] in {s[0], '.'}:
            # s不为空且p的第一个字符等于s的第一个字符或者等于.
            # 那么第一个字符才会匹配
            first_match = True
        else:
            first_match = False
        # A or B and C == A or (B and C)
        if len(p) > 1 and p[1] == '*':
            if self.isMatch(s, p[2:]):
                # 如果*代表0个字符的情况
                return True
            elif first_match and self.isMatch(s[1:], p):
                # 如果第一个就已经匹配了，就匹配下一个字符
                return True
            else:
                return False
        else:
            # 如果没有*出现的话，继续判断下一个字符
            return first_match and self.isMatch(s[1:], p[1:])


if __name__ == '__main__':
    s = 'aa'
    p = 'a*'
    res = Solution().isMatch(s, p)
    print(res)
