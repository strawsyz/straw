# 也是二分法
class Solution:
    def mySqrt(self, x: int) -> int:
        if x < 2:
            return x
        left, right = 2, x // 2 + 1
        while left <= right:
            mid = left + (right - left) // 2
            if mid * mid > x:
                right = mid - 1
            elif mid * mid < x:
                left = mid + 1
            else:
                return mid
        return right



if __name__ == '__main__':
    s = Solution()
    res = s.mySqrt(0)
    print(res)
