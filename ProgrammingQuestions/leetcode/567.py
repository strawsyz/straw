# 使用collections库计算
# 效率很低，但使用了collections库

class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        import collections
        # 计算每个字的个数
        count_dict = collections.Counter(s1)
        len_s1 = len(s1)
        left = 0
        right = len_s1
        len_s2 = len(s2) + 1
        while right < len_s2:
            if collections.Counter(s2[left:right]) == count_dict:
                return True
            left += 1
            right += 1
        return False
