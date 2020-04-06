def heap_sort(lst):
    def headad_just(arr, start, end):
        # start是根节点
        temp = arr[start]
        # 节点的是从0开始
        son = 2 * start + 1
        while son <= end:
            if son < end and arr[son] < arr[son + 1]:
                son += 1
            if temp < arr[son]:
                # 子节点大于父节点的话，交换子节点和父节点
                arr[start] = arr[son]
                start = son
                son = 2 * son + 1
        #  最后把最大的树放大输出
        arr[start] = temp

    n = len(lst)
    if n <= 1:
        return lst
    # 计算最后一个父节点的index
    root = n // 2 - 1
    # 循环调整每个子树中的关系
    while root >= 0:
        headad_just(lst, root, n - 1)
        root -= 1
    # 根节点，调整树的长度
    i = n - 1
    while i > 0:
        lst[0], lst[i] = lst[i], lst[0]
        headad_just(lst, 0, i - 1)
        i -= 1

    return lst


if __name__ == '__main__':
    lst = [1, 2, 34, 345, 235, 78]

    res = heap_sort(lst)
    print(res)

