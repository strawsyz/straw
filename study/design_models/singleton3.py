# 单例模式示例
# 一种基于共享状态的替代实现
# 虽然生成的对象不是同一个对象，但是里面的内容是一样的
# 不推荐单例的类有多级的继承

class Singleton:
    _state = {}

    def __new__(cls, *args, **kwargs):
        ob = super(Singleton, cls).__new__(cls, *args, **kwargs)
        ob.__dict__ = cls._state
        return ob


class Animal(Singleton):
    a = 1


class Dog(Animal):
    b = 2


if __name__ == '__main__':
    animal1 = Animal()
    animal2 = Animal()
    dog1 = Dog()
    dog2 = Dog()

    print(id(animal1) == id(animal2))
    print(id(dog1) == id(dog2))
    dog1.a = 10
    print(id(dog1.a) == id(dog2.a))
    print(dog1.a == dog2.a)
    print(id(dog1.b) == id(dog2.b))
    print(dog1.b == dog2.b)
