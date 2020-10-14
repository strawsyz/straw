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
        if getattr(self, attr_name, None) is not None:
            setattr(target, attr_name, getattr(self, attr_name))


def pretty_print(print_info: dict, column_names=None, use_pt=True):
    print_info = sorted(print_info)
    if use_pt:
        from prettytable import PrettyTable
        config_view = PrettyTable()
        if column_names is not None:
            config_view.field_names = column_names
            for key in print_info:
                for item in print_info[key]:
                    config_view.add_row(item)
        else:
            config_view.field_names = ["name", "value"]
            for key in print_info:
                config_view.add_row([key, print_info[key]])
        return config_view
    else:
        items = []
        if column_names is not None:
            items.append("\t".join(column_names))
        else:
            items.append("\t".join(["name", "value"]))

        for key in print_info:
            items.append("\t".join(print_info[key]))

        return "\n".join(items)
