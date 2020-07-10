# 访问者模式

class Vlist(list):
    def accept(self, visitor):
        visitor.visit_list(self)


class Vdict(dict):
    def accept(self, visitor):
        visitor.visit_dict(self)


class Printer:
    def visit_list(self, ob):
        print("list content")
        print(str(ob))

    def visit_dict(self, ob):
        print("dict content")
        print("dict keys: {}".format(",".join(ob.keys())))


def visit(visited, visitor):
    cls = visited.__class__.__name__
    meth = "visit_{}".format(cls)
    method = getattr(visitor, meth, None)
    if method is not None:
        method(visited)


if __name__ == '__main__':
    a_list = Vlist([1, 2, 4])
    a_list.accept(Printer())

    a_dict = Vdict({'one': 1, 'two': 2, 'three': 3})
    a_dict.accept(Printer())
    # 直接使用函数来实现访问者模式
    visit([1, 2, 4], Printer())

    visit({"one": 1, "two": 2, "three": 3}, Printer())
