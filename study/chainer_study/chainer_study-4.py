# Initial setup following http://docs.chainer.org/en/stable/tutorial/basic.html
import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

import matplotlib.pyplot as plt


# Defining your own neural networks using `Chain` class
class MyChain(Chain):
    def __init__(self):
        super(MyChain, self).__init__(
            # 第一个参数设为None，可以根据第一次输入的变量来确定他的大小
            l1=L.Linear(None, 30),
            l2=L.Linear(None, 30),
            l3=L.Linear(None, 1)
        )

    def __call__(self, x):
        h = self.l1(x)
        h = self.l2(F.sigmoid(h))
        return self.l3(F.sigmoid(h))


# Setup a model
model = MyChain()

model_save_path = 'mlp.model'
print('Loading model')
# --- use NPZ format ---
serializers.load_npz(model_save_path, model)
# --- use HDF5 format (need h5py library) ---
# %timeit serializers.load_hdf5(model_save_path, model)



# define target function
def target_func(x):
    """Target function to be predicted"""
    return x ** 3 - x ** 2 + x ** -1 + x


# create efficient function to calculate target_func of numpy array in element wise
target_func_elementwise = np.frompyfunc(target_func, 1, 1)

# define data domain [xmin, xmax]
xmin = -3
xmax = 3
# number of training data
sample_num = 20


# calculate new data from model (predict value)
x_test_data = np.array(np.random.rand(sample_num) * (xmax - xmin) + xmin)  # create 20
x_test = Variable(x_test_data.reshape(-1, 1).astype(np.float32))
y_test_data = model(x_test).data  # this is predicted value

# calculate target function (true value)
x_detail_data = np.array(np.arange(xmin, xmax, 0.1))
y_detail_data = target_func_elementwise(x_detail_data)

plt.clf()
# plot model predict data
plt.scatter(x_test_data, y_test_data, color='k', label='Model predict value')
# plot target function
plt.plot(x_detail_data, y_detail_data, label='True value')
plt.legend(loc='lower right')
plt.show()
