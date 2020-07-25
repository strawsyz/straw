import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.dataset import concat_examples
from chainer.training import extensions
import chainer.dataset
import chainer.datasets

from chainer.datasets import TupleDataset
import os
import numpy as np
import pandas as pd

# 创建自己的数据
DATA_DIR = './data/'


def black_box_fn(x_data):
    return np.sin(x_data) + np.random.normal(0, 0.1, x_data.shape)


# os.mkdir(DATA_DIR)

x = np.arange(-5, 5, 0.01)
t = black_box_fn(x)
# 这边故意将x和t换过来了，不然根本拟合不了
df = pd.DataFrame({'x': t, 't': x}, columns={'x', 't'})
df.to_csv(os.path.join(DATA_DIR, 'my_data.csv'), index=False)


class MyDataset(chainer.dataset.DatasetMixin):

    def __init__(self, path, root, mean, crop_size, random=True):
        # self.debug = debug
        # df = pd.read_csv(file_path)
        # self.data = df.values.astype(np.float32)
        # if self.debug:
        #     print('[DEBUG] data:\n{}'.format(self.data))
        self.base = chainer.datasets.LabeledImageDataset(path, root)
        self.mean = mean.astype('f')
        self.crop_size = crop_size
        self.random = random

    def __len__(self):
        return len(self.data)

    def get_example(self, i):
        # It reads the i-th image/label pair and return a preprocessed image.
        # It applies following preprocesses:
        # - Cropping (random or center rectangular)
        # - Random flip
        # - Scaling to [0, 1] value
        crop_size = self.crop_size

        image, label = self.base[i]
        _, h, w = image.shape
        import random
        if self.random:
            # Randomly crop a region and flip the image
            top = random.randint(0, h - crop_size - 1)
            left = random.randint(0, w - crop_size - 1)
            if random.randint(0, 1):
                image = image[:, :, ::-1]
        else:
            # Crop the center
            top = (h - crop_size) // 2
            left = (w - crop_size) // 2
        bottom = top + crop_size
        right = left + crop_size

        image = image[:, top:bottom, left:right]
        image -= self.mean[:, top:bottom, left:right]
        image *= (1.0 / 255.0)  # Scale to [0, 1]
        return image, label
        # x, t = self.data[i]
        # return [x], [t]


from chainer.datasets import TransformDataset

x = np.arange(10)
t = x * x - x

original_dataset = TupleDataset(x, t)


def transform_function(in_data):
    x_i, t_i = in_data
    new_t_i = t_i + np.random.normal(0, 0.1)
    return x_i, new_t_i


transformed_dataset = TransformDataset(original_dataset, transform_function)

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


# ================验证/测试的数据分离===================

# Load the dataset and separate to train data and test data
dataset = MyDataset('data/my_data.csv')
train_ratio = 0.7
train_size = int(len(dataset) * train_ratio)
# chainer.datasets.split_dataset_random函数将这个数据集分成70%的训练数据和30%的测试数据。
train, test = chainer.datasets.split_dataset_random(dataset, train_size, seed=3)
