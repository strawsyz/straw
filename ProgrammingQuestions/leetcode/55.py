from typing import List


class Solution:
    def canJump(self, nums: List[int]) -> bool:
        # 注意如果nums的长度为1，那么无论什么数字都应该返回True
        # 不用考虑最后一个数字，因为里面的循环判断确保了能达到当前值的后面的值
        for i in range(len(nums)-1):
            if nums[i] == 0:
                index = 1
                # 如果当前数字是0，就不能单单是到达当前数字，要能达到当前数字后面的数字
                while nums[i - index] <= index and i - index >= 0:
                    index += 1
                if i - index < 0:
                    return False
        return True


if __name__ == '__main__':
    s = Solution()
    nums = [0]
    res = s.canJump(nums)
    print(res)
