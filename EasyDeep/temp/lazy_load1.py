class LazyProxy(object):
    def __init__(self, cls, *args, **kwargs):
        self.__dict__['_cls'] = cls
        self.__dict__['_params'] = args
        self.__dict__['_kwargs'] = kwargs

        self.__dict__["_obj"] = None

    def __getattr__(self, item):
        if self.__dict__["_obj"] is None:
            self._init_obj()
        return getattr(self.__dict__['_obj'], item)

    def __setattr__(self, key, value):
        if self.__dict__['_obj'] is None:
            self._init_obj()
        setattr(self.__dict__['_obj'], key, value)

    def _init_obj(self):
        # create a instance
        self.__dict__['_obj'] = object.__new__(self.__dict__['_cls'])
        # init a new instance
        self.__dict__['_obj'].__init__(*self.__dict__['_params'],
                                       **self.__dict__['_kwargs'])


class LazyInit:

    def __new__(cls, *args, **kwargs):
        return LazyProxy(cls, *args, **kwargs)


class A(LazyInit):

    def __init__(self, x):
        print("Init A")
        self.x = 14 + x


if __name__ == '__main__':
    a = A(1)
    print("Go")
    print(a.x)
    print(a.x)
