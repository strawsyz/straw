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
x_data = np.array(np.random.rand(sample_num) * (xmax - xmin) + xmin)  # create 20
y_data = target_func_elementwise(x_data)

x_detail_data = np.array(np.arange(xmin, xmax, 0.1))
y_detail_data = target_func_elementwise(x_detail_data)



model_save_path = 'mlp.model'
optimizer_save_path = 'mlp.state'


# Defining your own neural networks using `Chain` class
class MyChain(Chain):
    def __init__(self):
        super(MyChain, self).__init__(
            # 第一个参数设为None，可以根据第一次输入的变量来确定他的大小
            l1=L.Linear(None, 30),
            l2=L.Linear(None, 30),
            l3=L.Linear(None, 1)
        )
improve
    def __call__(self, x):
        h = self.l1(x)
        h = self.l2(F.sigmoid(h))
        return self.l3(F.sigmoid(h))


# Setup a model
model = MyChain()
# Setup an optimizer
optimizer = chainer.optimizers.MomentumSGD()
optimizer.use_cleargrads()  # this is for performance efficiency
optimizer.setup(model)

# Init/Resume
resume = True
if resume:
    print('Loading model & optimizer')
    # --- use NPZ format ---
    serializers.load_npz(model_save_path, model)
    serializers.load_npz(optimizer_save_path, optimizer)
    # --- use HDF5 format (need h5py library) ---
    #%timeit serializers.load_hdf5(model_save_path, model)
    #serializers.load_hdf5(optimizer_save_path, optimizer)




x = Variable(x_data.reshape(-1, 1).astype(np.float32))
y = Variable(y_data.reshape(-1, 1).astype(np.float32))


def lossfun(x, y):
    loss = F.mean_squared_error(model(x), y)
    return loss


# this iteration is "training", to fit the model into desired function.
for i in range(300):
    # 使用数据不断的训练
    optimizer.update(lossfun, x, y)

    # above one code can be replaced by below 4 codes.
    # model.cleargrads()
    # loss = lossfun(x, y)
    # loss.backward()
    # optimizer.update()

y_predict_data = model(x_detail_data.reshape(-1, 1).astype(np.float32)).data

# Save the model and the optimizer
print('saving model & optimizer')



# --- use NPZ format ---
serializers.save_npz(model_save_path, model)
serializers.save_npz(optimizer_save_path, optimizer)

plt.clf()
plt.scatter(x_data, y_data, color='r')
plt.plot(x_detail_data, np.squeeze(y_predict_data, axis=1))
plt.show()
