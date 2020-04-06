# 最大堆排序
# 平均时间复杂度	O(nlog n)
# 最坏时间复杂度	O(nlog n)
# 最优时间复杂度	O(nlog n)
def heap_sort(lst):
    def head_adjust(arr, start, end):
        """堆调整
        arr是原来的数字
        start是开始调整的父节点
        end是arr的length-1，即数组下标的最大值
        """
        # 因为数组下标从0开始，2 * start + 1是左子节点
        son = 2 * start + 1
        # 不断的调整节点，使得子节点都下于父节点，直到到达最后一个
        while son <= end:
            # 如果右子节点没有越界（即存在），且要大于左节点，就把son移动到右子节点
            # 让父节点与右子节点相互交换
            if son < end and arr[son] < arr[son + 1]:
                son += 1
            # 如果父节点小于子节点
            if arr[start] < arr[son]:
                # 子节点大于父节点的话，交换子节点和父节点
                arr[start], arr[son] = arr[son], arr[start]
                # 设置新的开始节点和开始节点对应的左子节点
                start = son
                son = 2 * son + 1
            #  如果父节点大于两个子节点中比较大的那个节点
            #  就说明父节点大于两个节点
            #  说明这个堆已经是个最大堆了，可以跳出循环了
            #  由于最大堆的排序是从最后一个父节点开始的，所以不用担心下面的节点是否是个最大堆
            #  因为在循环到i层节点之前，i+1层就已经都是最大堆了
            else:
                break
        #  最后把最大的树放大输出
        # arr[start] = temp

    length = len(lst)
    # 特殊情况处理
    if length <= 1:
        return lst
    # 创建最大堆
    # 计算最后一个父节点(非叶子节点)的index
    root = length // 2 - 1
    # 循环调整每个子树中的关系
    while root >= 0:
        head_adjust(lst, root, length - 1)
        root -= 1
    # 堆排序
    # 根节点，调整树的长度
    i = length - 1
    while i > 0:
        # 将最大的值，即数组的一个数字（因为经过最大堆排序）。与最后一个数字交换
        lst[0], lst[i] = lst[i], lst[0]
        # 不管已经排好序的部分的数字。因为最大的数字已经被移动到了最后，所以不用管最后的部分数字
        head_adjust(lst, 0, i - 1)
        i -= 1

    return lst


if __name__ == '__main__':
    lst = [9, 2, 1, 7, 6, 8, 5, 3, 4]

    res = heap_sort(lst)
    print(res)
