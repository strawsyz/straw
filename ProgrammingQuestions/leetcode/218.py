import bisect


# 参考大佬的解法
# 使用扫描法

class Solution:
    def getSkyline(self, buildings: List[List[int]]) -> List[List[int]]:
        # 使用三元组记录所有的点
        # (x, is_right, height)
        points = []
        for l, r, pre_height in buildings:
            # 0表示是左边点
            points.append((l, False, pre_height))
            # 1表示时右边点
            points.append((r, True, pre_height))
        # 给元组排序
        points = sorted(points)
        # 记录高度
        heights = [0]
        # 记录结果点，
        results = []
        # 记录上一个点的信息
        pre_point = (-1, True, 0)
        for i, (x, is_right, height) in enumerate(points):
            # 循环所有的建筑
            # (-1,1,0)
            pre_x, pre_is_right, pre_height = pre_point
            if is_right:
                # 如果是右边节点
                if height == heights[-1] and height != heights[-2]:
                    # 当前位置的高度就是最高度，并且不等于第二高度
                    results.append([x, heights[-2]])
                heights.remove(height)
            else:
                # 如果是左边的点
                if height > heights[-1]:
                    # 如果这栋建筑物的高度要大于上一个建筑物
                    if x == pre_x:
                        # 如果点前建筑物左边的起点等于上一个点的左边的起点
                        if pre_is_right:
                            # 如果前一个点是右边点
                            # 直接加入新的轮廓点
                            results.append([x, height])
                        else:
                            # 如果前一个点是左边点
                            # 说明当前点被包裹在建筑里面了
                            # 上一个轮廓点的高度需要修改到新的高度
                            results[-1][1] = height
                    else:
                        # 如果和上一个x点的位置不相同
                        results.append([x, height])
                    # 当前点设置为上一个点
                    pre_point = points[i]
                # 插入新的高度
                bisect.insort(heights, height)

        return results
