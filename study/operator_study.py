import operator
from _functools import reduce


# 实现阶乘的功能
def fun1():
    list1 = filter(lambda x: x % 2, range(8))
    return reduce(operator.mul, list1)


if __name__ == '__main__':
    print(fun1())
    from operator import itemgetter

    data = [("dog", "animal", 10, 4, 2),
            ("cat", "animal", 6, 4, 2),
            ("mouse", "animal", 10, 4, 2)]
    a = itemgetter(1, 0)
    for i in data:
        print(a(i))

    # from collections import namedtuple
    #
    # latlong = namedtuple("latlong", "lat long")
    # te = latlong(1,2)
    # print(te.lat)
    # from operator import attrgetter
    # f = attrgetter("attr_name")
    # get  obj.attr_name
    # f(obj)

    # from operator import methodcaller
    # f = methodcaller("method_name")
    # print(f(obj))

    te = {"a": 1, "b": 2}
    te2 = {"a": 1, "b": 2}
    print(type(te))
    print(te.items() & te2.items())
    print(te.items() | te2.items())
    print(te.items() - te2.items())
    print(te.items() ^ te2.items())
