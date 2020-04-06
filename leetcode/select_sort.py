# 每次从列表中选出最小的数字放到最前面
# 时间复杂度O(n^2)。
# 空间复杂度O(1)。
# 不稳定
def select_sort(lst):
    nums = len(lst)
    if nums <= 1:
        return lst
    # 不用管最后一个数字，因为前面的数字排好之后，最后面的自然就是最大的数字了
    for i in range(0, nums - 1):
        temp = lst[i]
        min_index = i
        # 将第i个数字和其他的所有数字进行比较
        for j in range(i + 1, nums):
            if temp > lst[j]:
                temp = lst[j]
                min_index = j
        if i != min_index:
            lst[i], lst[min_index] = lst[min_index], lst[i]

    return lst

if __name__ == '__main__':
    te = [12313, 123, 5, 4, 32, 5643, 6, -1, 23, 14, 562, 34, 1, 5, -1, 23]
    res = select_sort(te)
    print(res)

