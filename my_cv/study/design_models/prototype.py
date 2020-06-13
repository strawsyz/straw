import copy


# 原型设计模式
class Prototype:
    _type = None
    _value = None

    def clone(self):
        pass

    def get_type(self):
        return self._type

    def get_value(self):
        return self._value


class TypeA(Prototype):
    """继承自原型类"""

    def __init__(self, number):
        self._type = "TypeA"
        self._value = number

    def clone(self):
        return copy.copy(self)


class TypeB(Prototype):
    def __init__(self, number):
        self._type = "TypeB"
        self._value = number

    def clone(self):
        return copy.copy(self)


class ObjectFactory:
    """对象工厂，所有的对象使用原型来制造"""
    __type1Value1 = None
    __type1Value2 = None
    __type2Value1 = None
    __type2Value2 = None

    @staticmethod
    def initialize():
        ObjectFactory.__type1Value1 = TypeA(1)
        ObjectFactory.__type1Value2 = TypeA(2)
        ObjectFactory.__type2Value1 = TypeB(1)
        ObjectFactory.__type2Value2 = TypeB(2)

    @staticmethod
    def getType1Value1():
        "get a value of the type"
        return ObjectFactory.__type1Value1.clone()

    @staticmethod
    def getType1Value2():
        "get a value of type"
        return ObjectFactory.__type1Value2.clone()

    @staticmethod
    def getType2Value1():
        return ObjectFactory.__type2Value1.clone()

    @staticmethod
    def getType2Value2():
        return ObjectFactory.__type2Value2.clone()


if __name__ == "__main__":
    ObjectFactory.initialize()
    instance = ObjectFactory.getType1Value1()
    print("%s: %s" % (instance.get_type(), instance.get_value()))

    instance = ObjectFactory.getType1Value2()
    print("%s: %s" % (instance.get_type(), instance.get_value()))

    instance = ObjectFactory.getType2Value1()
    print("%s: %s" % (instance.get_type(), instance.get_value()))

    instance = ObjectFactory.getType2Value2()
    print("%s: %s" % (instance.get_type(), instance.get_value()))
