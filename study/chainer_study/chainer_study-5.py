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

train, test = chainer.datasets.get_mnist()
print('len(train), type ', len(train), type(train))
print('len(test), type ', len(test), type(test))
print('train[0]', type(train[0]), len(train[0]))

print('train[0][0]', train[1][1].shape)
np.set_printoptions(threshold=np.inf)  # set np.inf to print all.
print(train[0][1])

base_dir = ''

# Load the MNIST dataset from pre-inn chainer method
train, test = chainer.datasets.get_mnist(ndim=1)
import os

ROW = 4
COLUMN = 5
for i in range(ROW * COLUMN):
    # train[i][0] is i-th image data with size 28x28
    image = train[i][0].reshape(28, 28)  # not necessary to reshape if ndim is set to 2
    plt.subplot(ROW, COLUMN, i + 1)  # subplot with size (width 3, height 5)
    plt.imshow(image, cmap='gray')  # cmap='gray' is for black and white picture.
    # train[i][1] is i-th digit label
    plt.title('label = {}'.format(train[i][1]))
    plt.axis('off')  # do not show axis value
plt.tight_layout()  # automatic padding between subplots
# save the image
plt.savefig(os.path.join(base_dir, 'mnist_plot.png'))
plt.show()
