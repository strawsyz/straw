class Solution:
    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        # 存储需要的字符
        # k：字符 v：个数
        need = {}

        for c in ransomNote:
            need[c] = need.get(c, 0) + 1
        # 需要匹配的字符类型的个数
        need_match = len(need)

        for c in magazine:
            res = need.get(c)
            if res is not None:
                if res > 0:
                    need[c] = res - 1
                    if res == 1:
                        need_match -= 1
                        if need_match == 0:
                            break
        return need_match == 0
