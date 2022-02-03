# 姑且能过，但不是什么聪明的解法，还需要优化

class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        length_needle, length_haystack = len(needle), len(haystack)
        if length_needle > length_haystack:
            return -1
        # 当needle是个空字符串的时候
        # 不管haystack是什么都返回0
        if length_needle == 0:
            return 0

        left = 0
        next_char = needle[0]
        # 找到第一个匹配的点的位置
        for c in haystack:
            if next_char != c:
                left += 1
            else:
                break
        # 滑动窗口的左右指针
        right = left
        # 一直移动右边指针，直到走到haystack的边界
        while right < length_haystack:
            if haystack[right] == next_char:
                right += 1
                if right - left == length_needle:
                    return left
                next_char = needle[right - left]
            else:
                # 如果下一个值和右指针所指的字符不相同
                # 重新开始匹配
                # 初始化下一个值为needle的第一个字符
                next_char = needle[0]
                left += 1
                # 找到下一个与needle的第一个字符相同的字符
                for c in haystack[left:]:
                    if next_char != c:
                        left += 1
                    else:
                        break
                right = left
        return -1


if __name__ == '__main__':
    haystack = "mississippi"
    needle = "issip"
    s = Solution()
    res = s.strStr(haystack, needle)
    print(res)
