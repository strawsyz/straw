# Initial setup following http://docs.chainer.org/en/stable/tutorial/basic.html
import numpy as np
import chainer
import six
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

import matplotlib.pyplot as plt


class MyChain(Chain):
    def __init__(self, n_units, n_out):
        super(MyChain, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, n_units)
            self.l2 = L.Linear(None, n_units)
            self.l3 = L.Linear(None, n_out)

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)


class SoftmaxCLassifier(Chain):
    def __init__(self, predictor):
        super(SoftmaxCLassifier, self).__init__(
            predictor=predictor
        )

    def __call__(self, x, t):
        # 在这里定义了损失函数的计算的话，就可以将此模型设置为优化器来训练
        y = self.predictor(x)
        self.loss = F.softmax_cross_entropy(y, t)
        self.accuracy = F.accuracy(y, t)
        return self.loss


# init
n_units = 50
model = MyChain(n_units, 10)
classifier_model = SoftmaxCLassifier(model)
# Setup an optimizer
optimizer = chainer.optimizers.Adam()
optimizer.setup(classifier_model)
# Pass the loss function (Classifier defines it) and its arguments
# optimizer.update(classifier_model, x, t)

# 如果不是使用gpu就把变量gpu设置为-1
gpu = 0
if gpu >= 0:
    chainer.cuda.get_device(gpu).use()  # Make a specified GPU current
    classifier_model.to_gpu()  # Copy the model to the GPU
#     cuda.cupy相当于numpy的gpu版本
xp = np if gpu < 0 else cuda.cupy

# training
# 乱序
# print(perm[:10])
# sum_accuracy = 0
# sum_loss = 0
batchsize = 16
train, test = chainer.datasets.get_mnist()
import time
EPOCH = 10
for _ in range(EPOCH):
    # ===========================training=============================
    start = time.time()
    N = len(train)
    perm = np.random.permutation(N)
    # evaluation
    sum_accuracy = 0
    sum_loss = 0
    for i in range(0, N, batchsize):
        x = chainer.Variable(xp.asarray(train[perm[i:i + batchsize]][0]))
        t = chainer.Variable(xp.asarray(train[perm[i:i + batchsize]][1]))

        # Pass the loss function (Classifier defines it) and its arguments
        optimizer.update(classifier_model, x, t)

        sum_loss += float(classifier_model.loss.data) * len(t.data)
        sum_accuracy += float(classifier_model.accuracy.data) * len(t.data)
    end = time.time()
    elapsed_time = end - start
    throughput = N / elapsed_time
    print('train mean loss={}, accuracy={}, throughput={} images/sec'.format(
        sum_loss / N, sum_accuracy / N, throughput))

# ===========================testing=============================
N_test = len(test)
# evaluation
sum_accuracy = 0
sum_loss = 0
for i in six.moves.range(0, N_test, batchsize):
    index = np.asarray(list(range(i, i + batchsize)))
    x = chainer.Variable(xp.asarray(test[index][0]))
    t = chainer.Variable(xp.asarray(test[index][1]))

    loss = classifier_model(x, t)
    sum_loss += float(loss.data) * len(t.data)
    sum_accuracy += float(classifier_model.accuracy.data) * len(t.data)

print('test mean loss={}, accuracy={}'.format(
    sum_loss / N_test, sum_accuracy / N_test))

# Save the model and the optimizer
import os
save_path = 'mnist/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
print('save the model')
serializers.save_npz('{}/classifier_mlp.model'.format(save_path), classifier_model)
serializers.save_npz('{}/mlp.model'.format(save_path), model)
print('save the optimizer')
serializers.save_npz('{}/mlp.state'.format(save_path), optimizer)

