class cached_property:
    def __init__(self, func):
        self.func = func

    def __get__(self, instance, cls):
        if instance is None:
            return self
        val = self.func(instance)
        setattr(instance, self.func.__name__, val)
        return val


def copy_attr(source, target):
    if hasattr(source, "__dict__"):
        for attr in source.__dict__:
            if attr is not "config_instance":
                setattr(target, attr, getattr(source, attr))
    else:
        for attr in source.__slots__:
            if attr is not "config_instance":
                setattr(target, attr, getattr(source, attr))


def copy_need_attr(self, target, attr_names):
    for attr_name in attr_names:
        setattr(target, attr_name, getattr(self, attr_name))
