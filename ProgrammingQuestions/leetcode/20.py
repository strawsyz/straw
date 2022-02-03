# 超级快！大概

class Solution:
    def isValid(self, s: str) -> bool:
        left_part = ["[", "{", "("]
        map = {"[": "]", "(": ")", "{": "}"}
        right_part = ["]", ")", "}"]
        # 已经拥有的左边的列表
        have_left_parts = []
        # 目标的右边部分
        target = None
        for c in s:
            if c in left_part:
                have_left_parts.append(c)
                # 下一个需要的右边部分
                target = map[c]
                continue
            if target:
                # 如果有目标
                if c == target:
                    # 如果等于目标值
                    have_left_parts.pop()
                    if have_left_parts != []:
                        # 如果还有剩余。就设置新的目标值
                        target = map[have_left_parts[-1]]
                    else:
                        target = None
                    continue
                elif c in right_part:
                    # 如果在目标值以外的右边范围内有值的话
                    return False
            elif c in right_part:
                # 如果没有目标值。但是有着右边部分，说明有问题
                return False
        if have_left_parts == []:
            # 如果没有需要补全的部分了，说明结束了
            return True
        else:
            return False


if __name__ == '__main__':
    s = Solution()
    res = s.isValid("([])]")
    print(res)
