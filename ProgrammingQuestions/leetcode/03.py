class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        if len(s) < 2:
            return len(s)
        if s.count(s[0]) == len(s):
            return 1
        s = list(s)
        longest_str = [s[0]]
        # 把第一个数字放入可能队列
        potentially_longest_string = [s[0]]
        index_new_start = 0
        for i in range(1, len(s)):
            if s[i] in potentially_longest_string:
                # 如果新的字符在正在计算的子串中，说明本串已经到头了
                if len(potentially_longest_string) > len(longest_str):
                    # 如果当前串
                    longest_str = potentially_longest_string
                index_new_start = s.index(s[i], index_new_start) + 1
                # 更新可能存在的最长的那串
                potentially_longest_string = s[index_new_start:i + 1]
            else:
                # 添加到最长的长度的字符串中去、
                potentially_longest_string.append(s[i])
        if len(potentially_longest_string) > len(longest_str):
            longest_str = potentially_longest_string
        return len(longest_str)


if __name__ == '__main__':
    input = "pwwkew"
    solution = Solution()

    output = solution.lengthOfLongestSubstring(input)
    print(output)
