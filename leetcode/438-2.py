# 能过，但是特别慢
class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        p_set = set(p)
        match = len(p_set)
        need = {}
        need_raw = {}
        for i in p:
            need[i] = need.get(i, 0) + 1
            need_raw[i] = need_raw.get(i, 0) + 1
        left = right = 0
        result = []
        while right < len(s):
            char = s[right]
            right += 1
            if char in p_set:
                # have[char] = have.get(char, 0) + 1
                # 如果在需要的列表中，对应的字符减去1个
                need[char] -= 1
                # 如果需要的字符是零了
                if need[char] == 0:
                    # 需要匹配的字符减去一个
                    match -= 1
                # 如果字符太多的话
                if need[char] == -1:
                    # 如果某个字符的个数超过了需求
                    # 移动左指针，缩小滑动窗口
                    while s[left] != char:
                        # 特定的need就会增加
                        if need[s[left]] == 0:
                            # 如果原本是0，由于移动左指针导致失去一个匹配的字符，要修改match的值
                            # 待匹配的字符种类增加一种
                            match += 1
                        need[s[left]] += 1
                        left += 1
                    left += 1
                    # 回复到正好足够的状态
                    need[char] = 0
                if match == 0:
                    result.append(left)
                    need[s[left]] += 1
                    match += 1
                    left += 1
            else:
                left = right
                import copy
                need = copy.deepcopy(need_raw)
                # need = need_raw
                match = len(p_set)
        return result
