from __future__ import print_function
import argparse

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer import serializers
import numpy as np
from chainer import reporter
from chainer.dataset import concat_examples

parser = argparse.ArgumentParser(description='Train custom dataset')
parser.add_argument('--batchsize', '-b', type=int, default=10,
                    help='Number of images in each mini-batch')
parser.add_argument('--epoch', '-e', type=int, default=50,
                    help='Number of sweeps over the dataset to train')
parser.add_argument('--gpu', '-g', type=int, default=-1,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--out', '-o', default='result',
                    help='Directory to output the result')
parser.add_argument('--resume', '-r', default='',
                    help='Resume the training from snapshot')
parser.add_argument('--unit', '-u', type=int, default=50,
                    help='Number of units')
args = parser.parse_args(['-g', '0'])

print('GPU: {}'.format(args.gpu))
print('# unit: {}'.format(args.unit))
print('# Minibatch-size: {}'.format(args.batchsize))
print('# epoch: {}'.format(args.epoch))
print('')

from chainer import reporter


class MyMLP(chainer.Chain):

    def __init__(self, n_units):
        super(MyMLP, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(n_units)  # n_in -> n_units
            self.l2 = L.Linear(n_units)  # n_units -> n_units
            self.l3 = L.Linear(n_units)  # n_units -> n_units
            self.l4 = L.Linear(1)  # n_units -> n_out

    def __call__(self, *args):
        # Calculate loss
        h = self.forward(*args)
        t = args[1]
        # 使用均方误差
        self.loss = F.mean_squared_error(h, t)
        # 第一个参数是被观察者，第二个参数是观察者
        reporter.report({'loss': self.loss}, self)
        return self.loss

    def forward(self, *args):
        # Common code for both loss (__call__) and predict
        x = args[0]
        h = F.sigmoid(self.l1(x))
        h = F.sigmoid(self.l2(h))
        h = F.sigmoid(self.l3(h))
        h = self.l4(h)
        return h

    def predict(self, *args):
        with chainer.using_config('train', False):
            with chainer.no_backprop_mode():
                return self.forward(*args)

    def predict_2(self, *args, batchsize=32):
        data = args[0]
        x_list = []
        y_list = []
        t_list = []
        for i in range(0, len(data), batchsize):
            x, t = concat_examples(data[i:i + batchsize])
            y = self.predict(x)
            y_list.append(y.data)
            x_list.append(x)
            t_list.append(t)
        x_array = np.concatenate(x_list)[:, 0]
        y_array = np.concatenate(y_list)[:, 0]
        t_array = np.concatenate(t_list)[:, 0]
        return x_array, y_array, t_array


# Set up a neural network to train
# Classifier reports softmax cross entropy loss and accuracy at every
# iteration, which will be used by the PrintReport extension below.
model = MyMLP(args.unit)

if args.gpu >= 0:
    chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
    model.to_gpu()  # Copy the model to the GPU

# Setup an optimizer
optimizer = chainer.optimizers.MomentumSGD()
optimizer.setup(model)

import pandas as pd


class MyDataset(chainer.dataset.DatasetMixin):

    def __init__(self, filepath, debug=False):
        self.debug = debug
        # Load the data in initialization
        df = pd.read_csv(filepath)
        self.data = df.values.astype(np.float32)
        if self.debug:
            print('[DEBUG] data: \n{}'.format(self.data))

    def __len__(self):
        """return length of this dataset"""
        return len(self.data)

    def get_example(self, i):
        """Return i-th data"""
        x, t = self.data[i]
        return [x], [t]


# Load the dataset and separate to train data and test data
dataset = MyDataset('data/my_data.csv')
train_ratio = 0.7
train_size = int(len(dataset) * train_ratio)
train, test = chainer.datasets.split_dataset_random(dataset, train_size, seed=13)

# 打乱数据顺序
train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
test_iter = chainer.iterators.SerialIterator(test, args.batchsize, repeat=False, shuffle=False)

# Set up a trainer
updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

# Evaluate the model with the test dataset for each epoch
trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))

# Dump a computational graph from 'loss' variable at the first iteration
# The "main" refers to the target link of the "main" optimizer.
trainer.extend(extensions.dump_graph('main/loss'))

# Take a snapshot at each epoch
# trainer.extend(extensions.snapshot(), trigger=(args.epoch, 'epoch'))
trainer.extend(extensions.snapshot(), trigger=(1, 'epoch'))

# Write a log of evaluation statistics for each epoch
trainer.extend(extensions.LogReport())

# Print selected entries of the log to stdout
# Here "main" refers to the target link of the "main" optimizer again, and
# "validation" refers to the default name of the Evaluator extension.
# Entries other than 'epoch' are reported by the Classifier link, called by
# either the updater or the evaluator.
trainer.extend(extensions.PrintReport(
    ['epoch', 'main/loss', 'validation/main/loss', 'elapsed_time']))

# Plot graph for loss for each epoch
if extensions.PlotReport.available():
    trainer.extend(extensions.PlotReport(
        ['main/loss', 'validation/main/loss'],
        x_key='epoch', file_name='loss.png'))
else:
    print('Warning: PlotReport is not available in your environment')
# Print a progress bar to stdout
trainer.extend(extensions.ProgressBar())

if args.resume:
    # Resume from a snapshot
    serializers.load_npz(args.resume, trainer)

# Run the training
# trainer.run()
# serializers.save_npz('{}/mymlp.model'.format(args.out), model)



# Set up a neural network to train
# Classifier reports softmax cross entropy loss and accuracy at every
# iteration, which will be used by the PrintReport extension below.
model = MyMLP(args.unit)

if args.gpu >= 0:
    chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
    model.to_gpu()  # Copy the model to the GPU

# Setup an optimizer
optimizer = chainer.optimizers.MomentumSGD()
optimizer.setup(model)
# Load the dataset and separate to train data and test data
dataset = MyDataset('data/my_data.csv')
train_ratio = 0.7
train_size = int(len(dataset) * train_ratio)
train, test = chainer.datasets.split_dataset_random(dataset, train_size, seed=3)
train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
test_iter = chainer.iterators.SerialIterator(test, args.batchsize, repeat=False, shuffle=False)

# Set up a trainer
updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

# Evaluate the model with the test dataset for each epoch
trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))

# Dump a computational graph from 'loss' variable at the first iteration
# The "main" refers to the target link of the "main" optimizer.
trainer.extend(extensions.dump_graph('main/loss'))

# Take a snapshot at each epoch
#trainer.extend(extensions.snapshot(), trigger=(args.epoch, 'epoch'))
trainer.extend(extensions.snapshot(), trigger=(1, 'epoch'))

# Write a log of evaluation statistics for each epoch
trainer.extend(extensions.LogReport())

# Print selected entries of the log to stdout
# Here "main" refers to the target link of the "main" optimizer again, and
# "validation" refers to the default name of the Evaluator extension.
# Entries other than 'epoch' are reported by the Classifier link, called by
# either the updater or the evaluator.
trainer.extend(extensions.PrintReport(
    ['epoch', 'main/loss', 'validation/main/loss', 'elapsed_time']))

# Plot graph for loss for each epoch
if extensions.PlotReport.available():
    trainer.extend(extensions.PlotReport(
        ['main/loss', 'validation/main/loss'],
        x_key='epoch', file_name='loss.png'))
else:
    print('Warning: PlotReport is not available in your environment')
# Print a progress bar to stdout
trainer.extend(extensions.ProgressBar())

if args.resume:
    # Resume from a snapshot
    serializers.load_npz(args.resume, trainer)

# Run the training
trainer.run()
serializers.save_npz('{}/mymlp.model'.format(args.out), model)