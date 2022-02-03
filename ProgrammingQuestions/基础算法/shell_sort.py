# 希尔排序的核心在于间隔序列的设定。既可以提前设定好间隔序列，也可以动态的定义间隔序列

def shell_sort(lst):
    nums = len(lst)
    if nums <= 1:
        return lst
    # 如果不加上1，就会有问题
    step = (nums // 2) + 1
    while step >= 1:
        for start in range(0, step):
            import math
            for i in range(math.ceil(nums / step)):
                index = start
                while index + i * step < nums - step:
                    # 从前向后
                    # 插入排序
                    if lst[index + i * step] > lst[index + i * step + step]:
                        lst[index + i * step], lst[index + step + i * step] = lst[index + step + i * step], lst[
                            index + i * step]
                    index += step
        step //= 2
    return lst


if __name__ == '__main__':
    te = [12313, 123, 5, 4, 32, 5643, 6, -1, 23, 14, 562, 34, 1, 5, -1, 23]
    res = shell_sort(te)
    print(res)
