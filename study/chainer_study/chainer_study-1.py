from chainer import Variable
import numpy as np

a = np.asarray([1, 2, 3], dtype=np.float32)
x = Variable(a)
# print(x)
# print(x.debug_print())


import chainer.links as L

in_size = 3  # input vector's dimension
out_size = 2  # output vector's dimension

linear_layer = L.Linear(in_size, out_size)  # L.linear is subclass of `Link`

"""linear_layer has 2 internal parameters `W` and `b`, which are `Variable`"""
# print('W: ', linear_layer.W.data, ', shape: ', linear_layer.W.shape)
# print('b: ', linear_layer.b.data, ', shape: ', linear_layer.b.shape)

# Force update (set) internal parameters
linear_layer.W.data = np.array([[1, 2, 3], [0, 0, 0]], dtype=np.float32)
linear_layer.b.data = np.array([3, 5], dtype=np.float32)

x0 = np.array([1, 0, 0], dtype=np.float32)
x1 = np.array([1, 1, 1], dtype=np.float32)

x = Variable(np.array([x0, x1], dtype=np.float32))
y = linear_layer(x)
print('W: ', linear_layer.W.data)
print('b: ', linear_layer.b.data)
print('x: ', x.data)  # input is x0 & x1
print('y: ', y.data)  # output is y0 & y1
# W:  [[ 0.01068367  0.58748239 -0.16838944]
