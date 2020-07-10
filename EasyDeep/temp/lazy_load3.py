import threading

lock = threading.Lock()


# 多线程下的懒加载

class Singleton(object):

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, "_instance"):
            try:
                lock.acquire()
                if not hasattr(cls, "_instance"):
                    cls._instance = super(Singleton, cls).__new__(cls, *args, **kwargs)
            finally:
                lock.release()
        return cls._instance


if __name__ == '__main__':
    print(Singleton() == Singleton())
