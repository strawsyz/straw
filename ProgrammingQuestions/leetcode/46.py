# 回溯法是一种暴力搜索的方式

# 使用了回溯法

# 回溯法：
# 判断当前的节点是否访问过
# 访问节点前，将标志设为1
# 访问完节点，将标志设为0

class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        # 存储所有的结果
        result_all = []
        length = len(nums)
        # 如果访问过这个节点了，就设为1
        visited = [0] * len(nums)

        def dfs(n, nums, result):
            # result是单次的结果
            # n是走到了哪一步的，已经决定了n和数字的排序
            if n == length:
                # 如果长度和nums一样长，说明nums的所有节点都已经比遍历过了
                # 将结果添加到result_all中
                result_all.append(result[:])
            else:
                # 遍历所有的节点
                for i in range(length):
                    # 判断当前的节点是否访问过了
                    if not visited[i]:
                        # 将当前节点放到结果中
                        result.append(nums[i])
                        # 将当前节点设为已经访问过了
                        visited[i] = 1
                        # 递归调用dfs
                        dfs(n + 1, nums, result)
                        # 回溯到上一步，返回到没有访问过当前节点的状态
                        # 把刚刚加入的nums[i]扔出去
                        result.pop()
                        visited[i] = 0

        dfs(0, nums, [])
        return result_all
