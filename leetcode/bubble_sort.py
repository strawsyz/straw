# 每次比较相邻的两个数字，大的放在右边，小的放在左边
# 每轮将最大那一个移动到最后面
# 时间复杂度O(n^2)。
# 空间复杂度O(1)。
def bubble_sort(lst):
    # 从小到大排序
    nums = len(lst)
    if nums <= 1:
        return lst
    for i in range(nums):
        for j in range(nums - i - 1):
            # print(j)
            if lst[j] > lst[j + 1]:
                lst[j + 1], lst[j] = lst[j], lst[j + 1]
    return lst


if __name__ == '__main__':
    te = [12313, 123, 5, 4, 32, 5643, 6, 456, 2, -1]
    res = bubble_sort(te)
    print(res)
