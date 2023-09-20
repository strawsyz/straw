# -*- coding: utf-8 -*-
# File  : DD3 地下迷宫.py
# Author: strawsyz
# Date  : 2023/9/18


def has_left(point):
    col_i = point[1] - 1
    if col_i >= 0 and col_i < m and map[point[0]][col_i] == 1 and have_done_map[point[0]][col_i] == 0:
        return True
    else:
        return False


def has_right(point):
    col_i = point[1] + 1
    if col_i >= 0 and col_i < m and map[point[0]][col_i] == 1 and have_done_map[point[0]][col_i] == 0:
        return True
    else:
        return False


def has_top(point):
    row_i = point[0] - 1
    if row_i >= 0 and row_i < n and map[row_i][point[1]] == 1 and have_done_map[row_i][point[1]] == 0:
        return True
    else:
        return False


def has_bottom(point):
    row_i = point[0] + 1
    if row_i >= 0 and row_i < n and map[row_i][point[1]] == 1 and have_done_map[row_i][point[1]] == 0:
        return True
    else:
        return False


# 不走过来的路，除非没路可走了
# 1. 向右，2.向左，3 向下
def next_step(point):
    # from_bottom, from_top, from_left, from_right = False
    no_way_flag = False
    # if len(steps) > 0:
    #     pre_point = steps[-1]
    # if (pre_point[0] - point[0]) == 1:
    #     from_bottom = True
    # elif (pre_point[0] - point[0]) == -1:
    #     from_top = True
    # else:
    #     if (pre_point[1] - point[1]) == 1:
    #         from_right = True
    #     else:
    #         from_left = True

    if has_top(point):
        if sum(used_P) + 3 > P:
            no_way_flag = True
        else:
            return [point[0] - 1, point[1]], 3
    elif has_right(point):
        if sum(used_P) + 1 > P:
            no_way_flag = True
        else:
            return [point[0], point[1] + 1], 1
    elif has_left(point):
        if sum(used_P) + 1 > P:
            no_way_flag = True
        else:
            return [point[0], point[1] - 1], 1
    elif has_bottom(point):
        return [point[0] + 1, point[1]], 0
    else:
        no_way_flag = True

    if no_way_flag:
        return None


# 1001
# 1101
# 0111
# 0011
import sys

# import numpy as np

inputs = []
for line in sys.stdin:
    a = line.split()
    inputs.append(a)

line = inputs[0]

n, m, P = [int(i) for i in line]

map = []
for input in inputs[1:]:
    map.append([int(i) for i in input])
# n, m, P = 4, 4, 10
# map = [[1, 0, 0, 1], [1, 1, 0, 1], [0, 1, 1, 1], [0, 0, 1, 1]]
steps = [[0, 0]]  # 记录当前正在行走的路
used_P = [0]  # 消耗的体力
have_done_map = [[0 for j in range(m)] for i in range(n)]  # 记录走过的路
have_done_map[0][0] = 1
step = [0, 0]

while True:
    step = next_step(step)
    if step is None:
        # last_step = steps[-1]
        # todo 恢复used_P , 恢复have used map?
        used_P = used_P[:-1]
        steps = steps[:-1]
        if len(steps) == 0:
            break
        step = steps[-1]
    else:
        step, step_used_P = step
        used_P.append(step_used_P)
        steps.append(step)
        have_done_map[step[0]][step[1]] = 1
        if step == [0, m - 1]:
            break

# print(steps)
if len(steps) == 0:
    print("Can not escape!")
else:
    for step in steps[:-1]:
        print(f"[{step[0]},{step[1]}]", end=",")
    step = steps[-1]
    print(f"[{step[0]},{step[1]}]")
