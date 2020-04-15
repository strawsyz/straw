# 使用动态规划法

class Solution:
    def longestPalindrome(self, s: str) -> str:
        length = len(s)
        # 特殊情况处理
        if length == 1:
            return s
        if length == 0:
            return ""
        # dp[i][j] 表示 从i到j能否组成回文
        dp = [[False for _ in range(length)] for _ in range(length)]
        max_len = 1
        index_start_char = 0
        # 每个字符自身就能组成回文
        for i in range(length):
            dp[i][i] = True

        for end in range(1, length):
            # 先选定最后一个字符的位置
            for start in range(end):
                # start从0开始
                if s[start] == s[end]:
                    # 如果第一个和最后一个字符相同
                    if end - start < 3:
                        # 如果长度小于等于3，说明能构成回文
                        # 实际长度是end-start+1<=3
                        # 所以是end-start<2
                        dp[start][end] = True
                    else:
                        # 如果大于3，根据回文去掉头尾两个字符后的字符串是否是回文来判断
                        dp[start][end] = dp[start + 1][end - 1]
                else:
                    # 不构成回文
                    dp[start][end] = False
                # 如果构成了回文
                if dp[start][end]:
                    # 判断当前长度是否是目前最长的,如果是就更新
                    cur_len = end - start + 1
                    if cur_len > max_len:
                        max_len = cur_len
                        index_start_char = start
        return s[index_start_char:index_start_char + max_len]


if __name__ == '__main__':
    input = "aaaabbbbbbbbbbccccccccccddddddddddeeeeeeeeeeffffffffffgggggggggghhhhhhhhhhiiiiiiiiiijjjjjjjjjjkkkkkkkkkkllllllllllmmmmmmmmmmnnnnnnnnnnooooooooooppppppppppqqqqqqqqqqrrrrrrrrrrssssssssssttttttttttuuuuuuuuuuvvvvvvvvvvwwwwwwwwwwxxxxxxxxxxyyyyyyyyyyzzzzzzzzzzyyyyyyyyyyxxxxxxxxxxwwwwwwwwwwvvvvvvvvvvuuuuuuuuuuttttttttttssssssssssrrrrrrrrrrqqqqqqqqqqppppppppppoooooooooonnnnnnnnnnmmmmmmmmmmllllllllllkkkkkkkkkkjjjjjjjjjjiiiiiiiiiihhhhhhhhhhggggggggggffffffffffeeeeeeeeeeddddddddddccccccccccbbbbbbbbbbaaaaaaaabbbbbbbbbbccccccccccddddddddddeeeeeeeeeeffffffffffgggggggggghhhhhhhhhhiiiiiiiiiijjjjjjjjjjkkkkkkkkkkllllllllllmmmmmmmmmmnnnnnnnnnnooooooooooppppppppppqqqqqqqqqqrrrrrrrrrrssssssssssttttttttttuuuuuuuuuuvvvvvvvvvvwwwwwwwwwwxxxxxxxxxxyyyyyyyyyyzzzzzzzzzzyyyyyyyyyyxxxxxxxxxxwwwwwwwwwwvvvvvvvvvvuuuuuuuuuuttttttttttssssssssssrrrrrrrrrrqqqqqqqqqqppppppppppoooooooooonnnnnnnnnnmmmmmmmmmmllllllllllkkkkkkkkkkjjjjjjjjjjiiiiiiiiiihhhhhhhhhhggggggggggffffffffffeeeeeeeeeeddddddddddccccccccccbbbbbbbbbbaaaa"
    solution = Solution()

    output = solution.longestPalindrome(input)
    print(output)
