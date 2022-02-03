# 快速排序

# 自己写的简单的快排
# def quick_sort(lst):
#     length = len(lst)
#     if length <= 1:
#         return lst
#     temp = lst[0]
#     smaller_numbers = []
#     bigger_numbers = []
#     res = []
#     for i in range(1, length):
#         if temp > lst[i]:
#             smaller_numbers.append(lst[i])
#         else:
#             bigger_numbers.append(lst[i])
#     if smaller_numbers != []:
#         res = res + quick_sort(smaller_numbers)
#     res = res + [temp]
#     if bigger_numbers != []:
#         res = res + quick_sort(bigger_numbers)
#     return res

# 《算法导论》中的快排程序
def quick_sort(array, l, r):
    if l < r:
        q = partition(array, l, r)
        quick_sort(array, l, q - 1)
        quick_sort(array, q + 1, r)
    return array

def partition(array, l, r):
    # 取数组最后边的数字作为哨兵
    x = array[r]
    # i标志小于x的一个数字的下标
    i = l - 1
    # 从左边开始循环
    for j in range(l, r):
        # 如果当前的数字小于哨兵
        if array[j] <= x:
            i += 1
            # 将当前数字移动到前面去，i标志小于x的一个数字的下标
            array[i], array[j] = array[j], array[i]
    array[i + 1], array[r] = array[r], array[i + 1]
    return i + 1


if __name__ == '__main__':
    res = quick_sort([1, 2, 2, 4, 6, 3, 45, 5, 6], 0, 8)
    print(res)
