# 单例模式示例
# 缺点不能再继承得到新的子类

class Singleton:
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '_instance'):
            orig = super(Singleton, cls)
            cls._instance = orig.__new__(cls, *args, **kwargs)
        return cls._instance


class Dog(Singleton):
    a = 1


if __name__ == '__main__':
    dog1 = Dog()
    dog2 = Dog()
    print(id(dog1) == id(dog2))
    print(id(dog1.a) == id(dog2.a))
