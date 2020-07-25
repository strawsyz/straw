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

x = np.arange(10)
t = x * x

data = TupleDataset(x, t)

print('data type: {}, len: {}'.format(type(data), len(data)))

# Get 1st, 2nd, 3rd data at the same time.
examples = data[0:4]

# print(examples)
# print('examples type: {}, len: {}'
#       .format(type(examples), len(examples)))

data_minibatch = concat_examples(examples)

# print(data_minibatch)
# print('data_minibatch type: {}, len: {}'
# .format(type(data_minibatch), len(data_minibatch)))
# 要将示例转换为小批量格式，可以在chainer.dataset中使用concat_examples函数。返回的数值格式是 ([x_array], [t array], ...)
x_minibatch, t_minibatch = data_minibatch
# Now it is array format, which has shape
# print('x_minibatch = {}, type: {}, shape: {}'.format(x_minibatch, type(x_minibatch), x_minibatch.shape))
# print('t_minibatch = {}, type: {}, shape: {}'.format(t_minibatch, type(t_minibatch), t_minibatch.shape))


from chainer.datasets import DictDataset

x = np.arange(10)
t = x * x

# To construct `DictDataset`, you can specify each key-value pair by passing "key=value" in kwargs.
data = DictDataset(x=x, t=t)

# print('data type: {}, len: {}'.format(type(data), len(data)))
# Get 3rd data at the same time.
example = data[2]

# print(example)
# print('examples type: {}, len: {}'
#       .format(type(example), len(example)))
#
# # You can access each value via key
# print('x: {}, t: {}'.format(example['x'], example['t']))


import os
# 可以使用ImageDataset类在每次创建小批量时从外存储器（例如硬盘）中打开图像。
from chainer.datasets import ImageDataset

# print('Current direcotory: ', os.path.abspath(os.curdir))

filepath = './data/images.dat'
image_dataset = ImageDataset(filepath, root='./data/images')

# print('image_dataset type: {}, len: {}'.format(type(image_dataset), len(image_dataset)))

# Access i-th image by image_dataset[i].
# image data is loaded here. for only 0-th image.
img = image_dataset[0]

# img is numpy array, already aligned as (channels, height, width),
# which is the standard shape format to feed into convolutional layer.
# print('img', type(img), img.shape)

# 它与ImageDataset类似，允许在运行时将图像文件从存储器加载到内存中。
# 不同之处在于它包含了标签信息，

from chainer.datasets import LabeledImageDataset

# print('Current direcotory: ', os.path.abspath(os.curdir))

filepath = './data/image_labels.dat'
labeled_image_dataset = LabeledImageDataset(filepath, root='./data/images')

print('labeled_image_dataset type: {}, len: {}'.format(type(labeled_image_dataset), len(labeled_image_dataset)))
# Access i-th image and label by image_dataset[i].
# image data is loaded here. for only 0-th image.
img, label = labeled_image_dataset[0]

print('img', type(img), img.shape)
print('label', type(label), label)