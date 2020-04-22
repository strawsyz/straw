# 归并排序
# 先将分成两部分，将对分开的两部分进行归并排序
# 采用分治的思想
# 总结，列表切片时要注意范围
# 归并排序是一种稳定的排序方法
# 代价是需要额外的内存空间。
# 稳定
def merge_sort(lst):
    def merge(arr, left, mid, right):
        # 存储临时数据
        temp = []
        left_i = left
        right_i = mid + 1
        while left_i <= mid and right_i <= right:
            # 比较左右序列的大小
            if arr[left_i] <= arr[right_i]:
                temp.append(arr[left_i])
                left_i += 1
            else:
                temp.append(arr[right_i])
                right_i += 1
        if left_i > mid and right_i <= right:
            # temp.append(arr[right_i:right])
            temp += arr[right_i:right + 1]
        if right_i > right and left_i <= mid:
            temp += arr[left_i: mid + 1]
        # print(left)
        # print(right)
        arr[left: right + 1] = temp
        # return temp

    right = len(lst) - 1
    if right == 0:
        return lst

    left = 0

    def sort(arr, left, right):
        if left >= right:
            return
        mid = (left + right) // 2
        sort(arr, left, mid)
        sort(arr, mid + 1, right)
        merge(arr, left, mid, right)

    sort(lst, left, right)
    return lst


if __name__ == '__main__':
    temp = [123, 783457, 24, 45, 2, 6, 2, 1]
    res = merge_sort(temp)
    print(res)
