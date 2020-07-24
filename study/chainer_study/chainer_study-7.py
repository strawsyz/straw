# Initial setup following http://docs.chainer.org/en/stable/tutorial/basic.html
import numpy as np
import chainer
import six
from chainer import cuda, Function, gradient_check, report, training, utils, Variable, computational_graph
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Chainer example: MNIST')
parser.add_argument('--initmodel', '-m', default='',
                    help='Initialize the model from given file')
parser.add_argument('--batchsize', '-b', type=int, default=100,
                    help='Number of images in each mini-batch')
parser.add_argument('--epoch', '-e', type=int, default=20,
                    help='Number of sweeps over the dataset to train')
parser.add_argument('--gpu', '-g', type=int, default=-1,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--out', '-o', default='result/2',
                    help='Directory to output the result')
parser.add_argument('--resume', '-r', default='',
                    help='Resume the training from snapshot')
parser.add_argument('--n_units', '-u', type=int, default=50,
                    help='Number of units')

train, test = chainer.datasets.get_mnist()
args = parser.parse_args(['-g', '0'])

batchsize = args.batchsize
n_epoch = args.epoch
N = len(train)  # training data size
N_test = len(test)  # test data size


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
n_units = args.n_units
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

if args.initmodel:
    print('Load model from', args.initmodel)
    serializers.load_npz(args.initmodel, classifier_model)
if args.resume:
    print('Load optimizer state from', args.resume)
    serializers.load_npz(args.resume, optimizer)
import os
if not os.path.exists(args.out):
    os.makedirs(args.out)
import time
for epoch in range(1, n_epoch + 1):
    print('epoch', epoch)

    # training
    perm = np.random.permutation(N)
    sum_accuracy = 0
    sum_loss = 0
    start = time.time()
    for i in range(0, N, batchsize):
        x = chainer.Variable(xp.asarray(train[perm[i:i + batchsize]][0]))
        t = chainer.Variable(xp.asarray(train[perm[i:i + batchsize]][1]))

        # Pass the loss function (Classifier defines it) and its arguments
        optimizer.update(classifier_model, x, t)

        if epoch == 1 and i == 0:
            with open('{}/graph.dot'.format(args.out), 'w') as o:
                g = computational_graph.build_computational_graph(
                    (classifier_model.loss,))
                o.write(g.dump())
            print('graph generated')

        sum_loss += float(classifier_model.loss.data) * len(t.data)
        sum_accuracy += float(classifier_model.accuracy.data) * len(t.data)
    end = time.time()
    elapsed_time = end - start
    throughput = N / elapsed_time
    print('train mean loss={}, accuracy={}, throughput={} images/sec'.format(
        sum_loss / N, sum_accuracy / N, throughput))

    # evaluation
    sum_accuracy = 0
    sum_loss = 0
    for i in six.moves.range(0, N_test, batchsize):
        index = np.asarray(list(range(i, i + batchsize)))
        x = chainer.Variable(xp.asarray(test[index][0]))
        t = chainer.Variable(xp.asarray(test[index][1]))
        with chainer.no_backprop_mode():
            # When back propagation is not necessary,
            # we can omit constructing graph path for better performance.
            # `no_backprop_mode()` is introduced from chainer v2,
            # while `volatile` flag was used in chainer v1.
            loss = classifier_model(x, t)
        sum_loss += float(loss.data) * len(t.data)
        sum_accuracy += float(classifier_model.accuracy.data) * len(t.data)

    print('test mean loss={}, accuracy={}'.format(
        sum_loss / N_test, sum_accuracy / N_test))
