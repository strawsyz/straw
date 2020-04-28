# 把数组a划分为n个大小相同子区间（桶），每个子区间各自排序，
# 最后合并。桶排序要求数据的分布必须均匀，不然可能会失效。
# 计数排序是桶排序的一种特殊情况，
# 可以把计数排序当成每个桶里只有一个元素的情况。

# 桶排序
# 待排序数组只能放整数
def bucket_sort(arr):
    # 初始化为数字中每个数字减去最小值作为坐标
    # 如果满足对应差值的数字存在的话，就在对应的坐标上加1
    buckets = [0] * (max(arr) - min(arr) + 1)
    min_a = min(arr)
    for i in range(len(arr)):
        # 当前值减去最小值对应的个数
        buckets[arr[i] - min_a] += 1
    result = []
    for i in range(len(buckets)):
        if buckets[i] != 0:
            # 如果在该坐标上有值
            # 坐标加上最小值，还原为换来的值，然后根据对应的个数添加到结果中
            result += [i + min_a] * buckets[i]
    return result