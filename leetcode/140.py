# 记忆搜索法
# 效率不高，但姑且能过
class Solution:
    def wordBreak(self, s, wordDict):
        # 初始化如下，用于后续计算
        memo = {len(s): ['']}

        def sentences(i):
            if i not in memo:
                # 如果之前没有保存对应的数据
                memo[i] = []
                for j in range(i + 1, len(s) + 1):
                    # 如果i到j个字符的部分在字典中存在
                    if s[i:j] in wordDict:
                        # 继续处理字符串的后面部分
                        # 由于可能存在多个结果，所以返回的是一个list
                        for tail in sentences(j):
                            # 由于是对字符j之后的数字进行处理，所以要都加上相同的头部
                            # (tail and " " + tail) 作用：
                            # 如果tail不为空，才加入进来，否则什么也不加入
                            memo[i].append(s[i:j] + (tail and " " + tail))

            return memo[i]

        return sentences(0)


if __name__ == '__main__':
    s = "catsanddog"
    wordDict = ["cat", "cats", "and", "sand", "dog"]
    res = Solution().wordBreak(s, wordDict)
    print(res)
