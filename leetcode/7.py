class Solution:
    def reverse(self, x: int) -> int:
        # 标志是否为负数
        is_nega = False
        if x < 0:
            is_nega = True
            x = -x
        res = []
        while x > 0:
            res.append(x % 10)
            x = x // 10
        length = len(res) - 1
        result = 0
        for i in range(length + 1):
            result += 10 ** length * res[i]
            length -= 1
        #  处理越界的结果
        if result > 2 ** 31 - 1:
            if is_nega and result == 2 ** 31:
                pass
            else:
                return 0
        if is_nega:
            return -result
        else:
            return result


if __name__ == '__main__':
    s = Solution()
    res = s.reverse(0)
    print(res)
