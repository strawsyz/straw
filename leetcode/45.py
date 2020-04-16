class Solution:
    def jump(self, nums: List[int]) -> int:
        # 保存列表的长度
        length = len(nums)
        # 如果列表小于等于1就直接返回0，因为不需要跳跃
        if length <= 1:
            return 0
        # 走一步就能走到的距离
        max_range = nums[0] + 1
        if length <= max_range:
            return 1
        last_range = 1
        step = 1
        while True:
            step += 1
            max = last_range
            for i in range(last_range, max_range):
                if nums[i] + i + 1 > max:
                    max = nums[i] + i + 1
            if max >= length:
                break
            last_range = max_range
            max_range = max
        return step

if __name__ == '__main__':
    s = Solution()
    res = s.jump([2, 3, 1, 1, 4])
    print(res)
