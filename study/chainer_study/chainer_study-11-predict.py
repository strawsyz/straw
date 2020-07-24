from __future__ import print_function
import argparse
import time

import numpy as np
import six
import matplotlib.pyplot as plt
import pandas as pd
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Chain, Variable, optimizers, serializers
from chainer import datasets, training, cuda, computational_graph
from chainer.dataset import concat_examples
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
        self.loss = F.mean_squared_error(h, t)
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

    def predict2(self, *args, batchsize=32):
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


# prepare dataset
dataset = MyDataset('data/my_data.csv')
train_ratio = 0.7
train_size = int(len(dataset) * train_ratio)
train, test = chainer.datasets.split_dataset_random(dataset, train_size, seed=3)

GPU = 0
n_units = 50
model_path = 'result/mymlp.model'
model = MyMLP(n_units)
batchsize = 16
if GPU >= 0:
    chainer.cuda.get_device(GPU).use()  # Make a specified GPU current
    model.to_gpu()  # Copy the model to the GPU
xp = np if GPU < 0 else cuda.cupy

serializers.load_npz(model_path, model)
# put data into GPU
x_list = []
y_list = []
t_list = []
for i in range(0, len(test), batchsize):
    x, t = concat_examples(test[i:i + batchsize])
    x_gpu = cuda.to_gpu(x)
    y = model.predict(x_gpu)
    y_list.append(y.data.tolist())
    x_list.append(x.tolist())
    t_list.append(t.tolist())

x_test = np.concatenate(x_list)[:, 0]
y_test = np.concatenate(y_list)[:, 0]
t_test = np.concatenate(t_list)[:, 0]

plt.figure()
plt.plot(x_test, t_test, 'o', label='test actual')
plt.plot(x_test, y_test, 'o', label='test predict')
plt.legend()
plt.savefig('predict_gpu.png')
plt.show()