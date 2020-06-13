p = (1, 2)

from collections import namedtuple

Point = namedtuple('Point', ['x', 'y'])
p = Point(1, 2)
print(p.x)
print(p.y)

print(isinstance(p, Point))
print(isinstance(p, tuple))

# 定义一个圆
Circle = namedtuple('Circle', ['x', 'y', 'r'])

print(Circle(1, 2, 3))

from collections import deque

# 队列
# 可以实现高效的插入和删除操作的双向列表
q = deque(['a', 'b', 'c'])
q.append('x')
q.appendleft('y')
print(q)
q.pop()
print(q)
q.popleft()
print(q)

from collections import defaultdict

# 给字典设置默认值
dd = defaultdict(lambda: 'N/A')
dd['key1'] = 'abc'
print(dd['key1'])
print(dd['key2'])

from collections import OrderedDict

# 给字典排序，
# 排序按照字段插入的顺序
d = dict([('a', 1), ('b', 2), ('c', 3)])
d['z'] = 1
d['y'] = 2
d['x'] = 3
print(d)
od = OrderedDict([('a', 1), ('b', 2), ('c', 3)])
print(od)
# od = OrderedDict()
od['z'] = 1
od['y'] = 2
od['x'] = 3
print(od.keys())


class LastUpdateOrderedDict(OrderedDict):
    def __init__(self, capacity):
        super(LastUpdateOrderedDict, self).__init__()
        self._capacity = capacity

    def __setitem__(self, key, value):
        # 新的键值是否已经存在
        contains_key = 1 if key in self else 0
        if len(self) - contains_key >= self._capacity:
            last = self.popitem(last=False)
            print("remove:", last)
        if contains_key:
            # 如果已经有之前的键了，就删去原来的键
            del self[key]
            print("set:", (key, value))
        else:
            print("add:", (key, value))

        OrderedDict.__setitem__(self, key, value)


from collections import Counter

c = Counter()
# c = defaultdict(lambda : 0)
for ch in 'programming':
    c[ch] = c[ch] + 1
print(c)
