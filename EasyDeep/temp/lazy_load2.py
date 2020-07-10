def lazy_property(func):
    attr_name = "_lazy_" + func.__name__

    @property
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            # 如果没有对应的属性
            print("seting...")
            # 就设置该属性
            setattr(self, attr_name, func(self))  # self->c, attr_name->_lazyarea, func->area()
        # 获得设置的属性
        return getattr(self, attr_name)

    return _lazy_property


class Circle(object):
    def __init__(self, radius):
        self.radius = radius

    @lazy_property
    def area(self):
        print('evalute')
        return 3.14 * self.radius ** 2


if __name__ == '__main__':
    c = Circle(4)
    print(c.__dict__)  # 输出：{'radius': 4}
    # 设置新的属性
    print(c.area)
    print(c.__dict__)
    print(c.area)
