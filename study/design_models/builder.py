# 构建器模式的优点
# 提供清晰的分离和独特的层次，可以在构建和表示由类创建的指定对象之间进行表示。
# 更好的控制所建模式的实现过程
# 提供改变对象内部表示的场景



class Director:
    """
    使用建造起来构建车
    """
    __builder = None

    def setBuilder(self, builder):
        self.__builder = builder

    def getCar(self):
        car = Car()

        # First goes the body
        body = self.__builder.get_body()
        car.set_body(body)

        # Then engine
        engine = self.__builder.get_engine()
        car.set_engine(engine)

        # And four wheels
        for _ in range(4):
            wheel = self.__builder.get_wheel()
            car.attach_wheel(wheel)

        return car


# The whole product
class Car:
    def __init__(self):
        self.__wheels = list()
        self.__engine = None
        self.__body = None

    def set_body(self, body):
        self.__body = body

    def attach_wheel(self, wheel):
        self.__wheels.append(wheel)

    def set_engine(self, engine):
        self.__engine = engine

    def specification(self):
        print("body: %s" % self.__body.shape)
        print("engine horsepower: %d" % self.__engine.horsepower)
        print("tire size: %d" % self.__wheels[0].size)


class Builder:
    def get_wheel(self): pass

    def get_engine(self): pass

    def get_body(self): pass


class JeepBuilder(Builder):
    """吉普车构造器"""

    def get_wheel(self):
        wheel = Wheel()
        wheel.size = 22
        return wheel

    def get_engine(self):
        engine = Engine()
        engine.horsepower = 400
        return engine

    def get_body(self):
        body = Body()
        body.shape = "SUV"
        return body


class Wheel:
    size = None


class Engine:
    horsepower = None


class Body:
    shape = None


if __name__ == "__main__":
    director = Director()
    jeepBuilder = JeepBuilder()
    print("Jeep")
    # 设置构造器
    director.setBuilder(jeepBuilder)
    # 构造吉普车
    jeep = director.getCar()
    jeep.specification()
    print("")
