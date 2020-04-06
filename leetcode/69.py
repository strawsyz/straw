class Solution:
    def mySqrt(self, x: int) -> int:
        # if x < 4:
        #     return 1
        # mid = x // 2
        left, right = 1, x // 2 + 1
        while left <= right:
            mid = (right + left) // 2
            if mid * mid < x:
                left = mid + 1
            elif mid * mid > x:
                right = mid - 1
            else:
                return mid
        return left - 1




if __name__ == '__main__':
    s = Solution()
    res = s.mySqrt(0)
    print(res)
