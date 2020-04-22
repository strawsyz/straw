# 假设前面i个数字是从小到大排序排好的
# 每次排序将第i+1个数字排到适当的位置去
# 时间复杂度O(n^2)
# 空间复杂度O(1)
# 稳定
def insert_sort(lst):
    nums = len(lst)
    if nums <= 1:
        return lst
    for i in range(1, nums):
        for j in range(i):
            if lst[i] < lst[j]:
                lst[j:i + 1] = lst[i:i + 1] + lst[j:i]
                # lst[i], lst[i - 1] = lst[i - 1], lst[i]
                break
    return lst


if __name__ == '__main__':
    te = [12313, 123, 5, 4, 32, 5643, 6, 456, 2]
    res = insert_sort(te)
    print(res)
