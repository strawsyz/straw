# 使用滑动窗口

class Solution:
    def minWindow(self, s: str, t: str) -> str:
        # match表示已经满足要求的字符类型的数量
        left, right, match = 0, 0, 0
        # need储存需要的字符及个数
        # have存储拥有的字符及个数
        need, have = {}, {}
        result = ""
        min_result_len = len(s) + 1
        for x in t:
            need[x] = need.get(x, 0) + 1
        tset = set(t)
        # 需要匹配的字符数量
        need_match = len(tset)
        while right < len(s):
            # 如果右指针所指的字符在需要的字符类型之内
            if s[right] in tset:
                # 将相应的字符的已经拥有的数量加一
                have[s[right]] = have.get(s[right], 0) + 1
                # 如果字符拥有的数量等于需要的数量
                if have[s[right]] == need[s[right]]:
                    # 标志一类字符已经足够了
                    match += 1
            # 右指针向右移动一步
            right += 1
            # 如果所有需要的字符类型全都满足了，就将开始缩小窗口
            # 保持右指针不动，不断向右移动左指针，直到刚好满足需要的字符类型的数量
            while match == need_match:
                # 如果新窗口小于最小子串长度
                if right - left < min_result_len:
                    # 更新结果
                    result = s[left:right]
                    # 更新最小子串长度
                    min_result_len = len(result)
                # 如果左指针所指的字符是需要的字符类型
                if s[left] in tset:
                    # 在拥有的字符中，减去相应的字符个数
                    have[s[left]] = have.get(s[left], 0) - 1
                    # 如果拥有的字符个数小于需要的字符个数。就减少一个match的值
                    if have[s[left]] < need[s[left]]:
                        match -= 1
                left += 1
        return result
