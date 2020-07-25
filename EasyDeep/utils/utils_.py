# 调用之后只执行一次
class cached_property:
    def __init__(self, func):
        self.func = func

    def __get__(self, instance, cls):
        if instance is None:
            return self
        val = self.func(instance)
        setattr(instance, self.func.__name__, val)
        return val
        # value = instance.__dict__[self.func.__name__] = self.func(instance)
        # return value


def copy_attr(source, target):
    if hasattr(source, "__dict__"):
        for attr in source.__dict__:
            setattr(target, attr, getattr(source, attr))
    else:
        for attr in source.__slots__:
            setattr(target, attr, getattr(source, attr))
    # todo 复制完之后删去配置文件占用的空间


class Test(object):
    def __init__(self, value):
        self.value = value

    @cached_property
    def display(self):
        # create expensive object
        print("some complicated compute here")
        return self.value


class A:
    c = "c"

    def __init__(self):
        self.a = "A"
        self.b = "B"


class B:
    pass


if __name__ == '__main__':
    # t=Test(1000)
    # print(t.__dict__)
    # print(t.display)
    # print(t.__dict__)
    # print(t.display)
    a = A()
    b = B()
    copy_attr(a, b)
    print(b.c)
