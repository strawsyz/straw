from typing import List


class Solution:
    def findClosestElements(self, arr: List[int], k: int, x: int) -> List[int]:
        left = 0
        length = len(arr)
        right = length - 1
        while left <= right:
            mid = (left + right) // 2
            if arr[mid] < x:
                left = mid + 1
            elif arr[mid] > x:
                right = mid - 1
            else:
                left = mid
                right = mid - 1
                break
        res = []
        count = 0
        for _ in range(k):
            if abs(arr[left] - x) < abs(arr[right] - x):
                res.append(left)
                if left == 0:
                    break
                else:
                    left = left - 1
            elif abs(arr[left] - x) > abs(arr[right] - x):
                res.append(right)
                if right == length - 1:
                    break
                else:
                    right = right + 1
            else:
                if left == right:
                    res.append(left)
                    if left == 0:
                        right = 1
                    elif right == length - 1:
                        left = right - 1

            count += 1

        if left == 0:
            for _ in range(k - count - 1):
                right
            right += 1
            res.append(right)

        elif right == length - 1:
            for _ in range(k - count - 1):
                left -= 1
                res.append(left)


        result = []
        for i in res:
            result.append(arr[i])

        return result

if __name__ == '__main__':
    s = Solution()
    arr = [1, 2, 3, 4]
    k = 3
    x = 1
    res = s.findClosestElements(arr, k, x)
    print(res)
