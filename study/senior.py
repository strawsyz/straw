def square(x):
    return x ** 2


res = map(square, [1, 2, 3, 4, 5])
print(list(res))

import pymongo


# 上下文管理
class Operation(object):
    def __init__(self, dataabase, host="localhost", port=12312):
        self._db = pymongo.MongoClient(host, port)[dataabase]

    def __enter__(self):
        return self._db

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._db.connection.disconnect()


# with Operation(dataabase="test") as db:
#     print(db.test.find_one())

# 使用contextlib
import contextlib


@contextlib.contextmanager
def operation(database, host="localhost", port=123213):
    _db = pymongo.MongoClient(host, port)[dataabase]
    yield _db
    _db.connection.disconnect()


# 使用__all__控制允许被引用的内容
__all__ = ["add", "x"]


# 自动指定子目录
# 写在包的__init__.py文件中
# from os.path import abspath, join
#
# subdirs = lambda *dirs: [abspath(
#     join(__path__[0], sub)) for sub in dirs]
# __path__ = subdirs("a", "b")


# 限制类实例绑定数据，大量数据时减少内存
class User:
    __slots__ = ('name', 'age')

    def __init__(self, name, age):
        self.name = name
        self.age = age


u = User("Dong", 28)
res = hasattr(u, '__dict__')
print(res)


# res.title = 213


# @cached_property
# 主要实现的功能是，被装饰的类实例方法在第一次调用后，
# 会把值缓存下来，下次再调用会直接从 __dict__ 取结果，
# 避免了多次计算；你可以参考下面的代码实现这个装饰器。

# 定义基类
#
class HelloMeta(type):
    def __new__(cls, name, bases, attrs):
        def __init__(self, func):
            cls.func = func

        def hello(cls):
            print("hello world")

        t = type.__new__(cls, name, bases, attrs)
        t.__init__ = __init__
        t.hello = hello
        # 最后返回要穿件的类
        return t

        # todo 带理解
        # class Hello(object):
        #     __metaclass__ = HelloMeta
        #
        # h = HelloMeta('sdas',(2,4),lambda x: x + 1)
        # h.hello()


        # 设置默认参数必须指向一个不可变类型

        # 闭包是迟绑定，
