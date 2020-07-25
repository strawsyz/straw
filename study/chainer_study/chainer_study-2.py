from chainer import Variable
import numpy as np
from chainer import Chain, Variable
import chainer.links as L


# Defining your own neural networks using `Chain` class
class MyChain(Chain):
    def __init__(self):
        super(MyChain, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(2, 2)
            self.l2 = L.Linear(2, 1)

    def __call__(self, x):
        h = self.l1(x)
        return self.l2(h)


x = Variable(np.array([[1, 2], [3, 4]], dtype=np.float32))

model = MyChain()
y = model(x)

print('x: ', x.data)  # input is x0 & x1
print('y: ', y.data)  # output is y0 & y1
