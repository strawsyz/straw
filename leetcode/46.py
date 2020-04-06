# 递归一定发生回溯
# 回溯法是一种暴力搜索的方式

# 使用了回溯法
# 回溯法的模板
# 判断当前的节点是否访问过
# 访问节点前，将标志设为1
# 访问完节点，将标志设为0

class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        result_all = [[]]
        visited = [0] * len(nums)

        def dfs(n, nums, result):
            # result是单次的结果
            #     n是走到了哪一步的，已经决定了n和数字的排序
            if n == len(nums):
                result_all.append(result[:])
                return

            for i in range(len(nums)):
                # 判断当前的节点是否访问过了
                if visited[i] == 1:
                    continue
                result.append(nums[i])
                visited[i] = 1
                dfs(n + 1, nums, result)
                # 回溯上一步
                result.pop()
                visited[i] = 0

        dfs(0, nums, [])
        return result_all
