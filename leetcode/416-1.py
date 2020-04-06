from typing import List


class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        all_sum = sum(nums)
        if all_sum % 2 != 0:
            return False
        target = all_sum // 2
        dp = [[0] * (target + 1) for _ in range(len(nums))]
        # 给矩阵目标为0的那个位置全都设为1，因为无论是怎样的数组都能构成数字0
        for i in range(len(nums)):
            dp[i][0] = 1
        # 如果数组的第一个数字小于最终目标，就用这个数字
        if nums[0] <= target:
            dp[0][nums[0]] = 1
        # 先循环除了第一个数字的其他数字
        for i in range(1, len(nums)):
            # 设num为第二个数字
            cur_num = nums[i]
            # 因为第一列都已经、设为1了，所以从第二列开始循环
            for j in range(1, target + 1):
                # 如果目标值小于当前的数字，说明无法添加当前的数字
                if cur_num > j:
                    # 直接把上一行的结果抄下来，因为当前增加的这个数字没有用，在不在都一样
                    # 也就说明和上一行（该数字不在的时候）一样
                    dp[i][j] = dp[i - 1][j]
                    continue
                # 如果当前值小于等于目标值
                # 上一行，且目标值和当前值一致的时候，如果上一行可以做到，这一行也必然可以
                # 除此以外，上一行，目标值等于当前目标值减去当前值，如果能做到的话，这一样也必然能做到
                dp[i][j] = dp[i - 1][j] or dp[i - 1][j - cur_num]
        # 最后一行最后一个列的值为0表示，没有结果。如果是1表示有结果
        # 最后一行最后一个列的值表示能否用所有的数字中的特定一组数字来表示target
        return dp[-1][-1]


if __name__ == '__main__':
    s = Solution()
    input = [1, 5, 11, 5]
    res = s.canPartition(input)
    print(res)
