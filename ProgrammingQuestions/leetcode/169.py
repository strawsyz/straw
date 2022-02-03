# 参考官方解法
# Python的每个对象都分为可变和不可变，主要的核心类型中，数字、字符串、元组是不可变的，列表、字典是可变的。

# 要考虑到各种斜率，不单单是四个方向
class Solution:
    def maxPoints(self, points: List[List[int]]) -> int:
        def max_points_on_a_line_containing_point_i(i):
            """
            计算经过点i的直线上点最多的线
            """

            def add_line(i, j, max_num_points, n_duplicate_point_i):
                """
                经过点i和点j的直线上，所有的点已经重复的点i的数量
                """
                x1, y1 = points[i][0], points[i][1]
                x2, y2 = points[j][0], points[j][1]
                # 如果点i和点j是相同的点
                if x1 == x2 and y1 == y2:
                    # 增加重复点的数量
                    n_duplicate_point_i += 1
                elif y1 == y2:
                    # 如果连个点的y相同，但是x坐标不相同，说明这是一条水平线
                    # 关于nonlocal
                    # 如果在内部函数中只是仅仅读外部变量，可以不在此变量前加nonlocal
                    # 如果在内部函数中尝试进行修改外部变量，且外部变量为不可变类型，
                    # 则需要在变量前加nonlocal，如果变量为可变类型，则不需要添加nonlocal
                    # 由于数字是不可变类型，所以需要添加nonlocal
                    nonlocal horisontal_lines
                    # 增加水平线的数量
                    horisontal_lines += 1
                    max_num_points = max(horisontal_lines, max_num_points)
                else:
                    # 如果不是水平线，就计算斜率
                    slope = (x1 - x2) / (y1 - y2)
                    # 如果没有斜率就设置新的斜率，
                    # 新斜率上经过的点的数量是两点
                    lines[slope] = lines.get(slope, 1) + 1
                    max_num_points = max(lines[slope], max_num_points)
                # 返回经过点i，j的线上最多点的数量，已经目前发现的重复的点i的数量
                return max_num_points, n_duplicate_point_i

            # lines记录通过点i的直线
            lines, horisontal_lines = {}, 1
            # 最少也会经过一点
            max_num_points = 1
            # 重复的点i的数量
            n_duplicate_point_i = 0
            for j in range(i + 1, n_points):
                # 每个点只要检测它和排在它后面的点的连线就可以了，因为每计算一个点，会把当前点的结果保存下来
                max_num_points, n_duplicate_point_i = add_line(i, j, max_num_points, n_duplicate_point_i)
            # 重复的点i也看做在线上的点
            return max_num_points + n_duplicate_point_i

        n_points = len(points)
        if n_points < 3:
            # 如果只有两个点，那么连接这两个点就可以了
            return n_points
        # 只要有两个以上的点，就至少能通过两个点
        result = 2
        for i in range(n_points - 1):
            # 循环每一个点
            result = max(max_points_on_a_line_containing_point_i(i), result)
        return result
