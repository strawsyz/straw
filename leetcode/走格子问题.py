# 从左上的出发
# 在m x n 的棋盘上走上下左右走
# 不能走走过的格子
# 要填满所有的格子。共有几种走法

# 暴力解法

if __name__ == '__main__':
    m = 4  # row
    n = 5  # col


    def is_over(row, col):
        if row < 0 or row > m - 1:
            return True
        elif col < 0 or col > n - 1:
            return True


    now = [0, 0]
    visited = [[0 for _ in range(n)] for _ in range(m)]
    visited[0][0] = 1
    moves = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    num_visited = 1
    result = 0


    def main(now, num_visited):
        result = 0
        if num_visited == m * n:
            return 1
        flag = False
        for move in moves:
            next_step = [now[0] + move[0], now[1] + move[1]]
            if is_over(next_step[0], next_step[1]):
                continue
            else:
                # 下一步还没被访问过
                if visited[next_step[0]][next_step[1]] == 0:
                    flag = True
                    # 走到下一步
                    # next_step = next_step
                    # 把当前设置为已经访问过了
                    visited[next_step[0]][next_step[1]] = 1
                    # 增加一个已访问的个数
                    num_visited += 1
                    # 将当前所在的位置和之前已经走过的路径，还有目前位置的结果数量传到函数去
                    # 加上目前已经有的步数
                    result += main(next_step, num_visited)
                    # 恢复到没有遍历的状态
                    # 已经访问的个数减去1
                    num_visited -= 1
                    # 将当期的位置设置为没有访问过
                    visited[next_step[0]][next_step[1]] = 0
        return result

    result = main(now, num_visited)
    print(result)
