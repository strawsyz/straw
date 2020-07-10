# 观察者模式

class Event(object):
    _observers = []

    def __init__(self, subject):
        self.subject = subject

    @classmethod
    def register(cls, observer):
        if observer not in cls._observers:
            cls._observers.append(observer)

    @classmethod
    def unregister(cls, observer):
        if observer in cls._observers:
            cls._observers.remove(observer)

    @classmethod
    def notify(cls, subject):
        event = cls(subject)
        for observer in cls._observers:
            observer(event)


class WriteEvent(Event):
    def __repr__(self):
        return "WriteEvent"


# 函数类型的观察者
def log(event):
    print("{} is written".format(event.subject))


# 类类型的观察者
class AnotherObserver(object):
    def __call__(self, event):
        print("{} tell me".format(event))


if __name__ == '__main__':
    # 注册观察者
    WriteEvent.register(log)
    WriteEvent.register(AnotherObserver())
    # 触发事件
    WriteEvent.notify("a given file")
