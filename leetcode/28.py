# 姑且能过，但不是什么聪明的解法，还需要优化

class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        length_needle, length_haystack = len(needle), len(haystack)
        if length_needle > length_haystack:
            return -1
        if length_needle == 0:
            return 0
        left = 0
        next_char = needle[0]
        for c in haystack:
            if next_char != c:
                left += 1
            else:
                break
        right = left
        while right < len(haystack):
            if haystack[right] == next_char:
                right += 1
                if right - left == length_needle:
                    return left
                next_char = needle[right - left]
            else:
                next_char = needle[0]
                left += 1
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
