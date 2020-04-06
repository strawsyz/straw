class Solution:
    def convertToBase7(self, num: int) -> str:
        i = 0
        if num == 0:
            return '0'
        result = ''
        is_negative = num < 0
        if is_negative:
            num = -num

        while num > 0:
            result = str(num % 7) + result
            num //= 7
        if is_negative:
            result = '-' + result
        return result


if __name__ == '__main__':
    s = Solution()
    res = s.convertToBase7(100)
    print(res)
