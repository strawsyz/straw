class Solution:
    def kSimilarity(self, A: str, B: str) -> int:
        # 字符串A和B
        def bfs(A_part):
            i = 0
            while A_part[i] == B[i]:
                i += 1
            for j in range(i + 1, len(A_part)):
                if A_part[j] == B[i]:
                    # 如果找到了相同的字
                    # 将A的第i个字符和第j个字符交换后得到的字符串返回
                    yield A_part[:i] + A_part[j] + A_part[i + 1:j] + A_part[i] + A_part[j + 1:]

        # 存储交换字符之后A和交换的次数
        memo = [(A, 0)]
        # 由于字典数据复无法一边遍历一边修改数据
        # 为了加快遍历速度，维护一个哈希表
        visited = {A}
        for A_part, times in memo:
            # 在遍历q的时候，将新的数据append到q中，之后遍历的之后也会遍历到新添加的数据
            if A_part == B:
                return times
            for y in bfs(A_part):
                # 交换之后获得的字符串
                # 遍历所有符合条件的交换
                # 防止在visited和queue中放入重复的数据，要进行判断
                if y not in visited:
                    visited.add(y)
                    # 将交换的结果放入字典，并增加对应对应的次数
                    memo.append((y, times + 1))


if __name__ == '__main__':
    A = "sdasd"
    B = "ssdad"
    res = Solution().kSimilarity(A, B)
    print(res)
