from functools import partial


# 偏函数(functools.partial)


def add(*args, **kwargs):
    # 打印位置参数
    for n in args:
        print(n)
    print("-" * 20)
    # 打印关键字参数
    for k, v in kwargs.items():
        print('%s:%s' % (k, v))


add_partial = partial(add, 10, k1=10, k2=20)
if __name__ == '__main__':
    # add(1, 2, 3, v1=10, v2=20)
    add_partial(1, 2, 3, k3=20)
