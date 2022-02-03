from typing import List


#  暴力解法，会超时

class Solution:
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        n_row = len(matrix)
        n_col = len(matrix[0])

        def get_moves(x, y):
            moves = []
            if x != 0:
                moves.append([x - 1, y])
            if x != n_row - 1:
                moves.append([x + 1, y])
            if y != 0:
                moves.append([x, y - 1])
            if y != n_col - 1:
                moves.append([x, y + 1])
            return moves

        def get_bigger_moves(moves, val):
            next_moves = []
            for move in moves:
                row, col = move[0], move[1]
                if val < matrix[row][col]:
                    next_moves.append(move)
            return next_moves

        def dfs(next_moves):
            length = 0
            length += 1
            long = 0
            # 循环下一步能够走的位置
            for move in next_moves:
                row, col = move[0], move[1]
                # 新位置的值
                val = matrix[row][col]
                # 找到更加大的位置
                bigger_moves = get_bigger_moves(get_moves(row, col), val)
                if len(bigger_moves) > 0:
                    # 如果有下一步走的位置
                    temp = dfs(bigger_moves)
                else:
                    temp = 1
                if temp > long:
                    long = temp
            length += long

            return length

        longest_length = 1
        # 循环列表的每个位置
        for row in range(n_row):
            for col in range(n_col):
                # 判断周围的数字中是否有比当前数字更小的
                has_smaller = False
                # 当前值
                val = matrix[row][col]
                next_moves = []
                # 遍历所有能走的位置
                for move in get_moves(row, col):
                    row_, col_ = move[0], move[1]

                    if matrix[row_][col_] < val:
                        # 如果周围有更小的数字
                        # 说明当前位置不能作为起点
                        has_smaller = True
                        break
                    elif matrix[row_][col_] > val:
                        # 如果有比当前值大的，可以作为下一个步走的位置
                        next_moves.append(move)
                # 如果点的周围没有更大的点，或者有比当前点更小的点
                if has_smaller or len(next_moves) == 0:
                    continue
                temp = dfs(next_moves)
                if longest_length < temp:
                    longest_length = temp
        return longest_length


if __name__ == '__main__':
    res = Solution().longestIncreasingPath([[9, 9, 4], [6, 6, 8], [2, 1, 1]])
    print(res)
