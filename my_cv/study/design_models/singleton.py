# 单例模式示例

class Singleton:
    __instance = None

    @staticmethod
    def get_instance():
        if Singleton.__instance is None:
            Singleton()
        return Singleton.__instance

    def __init__(self):
        if Singleton.__instance is not None:
            # 由于是单例模式，所以不能创建两个
            raise Exception("This class is a singleton!")
        else:
            Singleton.__instance = self


if __name__ == '__main__':
    s1 = Singleton.get_instance()
    s2 = Singleton.get_instance()
    print(id(s1) == id(s2))
    s3 = Singleton()
    # s4 = Singleton()
    # print(s3 is s4)
