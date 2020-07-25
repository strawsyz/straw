from __future__ import print_function
import os
import matplotlib.pyplot as plt
from chainer.training import extensions
import chainer
import chainer.functions as F
import chainer.links as L

CIFAR10_LABELS_LIST = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]

# 获得数据
train, test = chainer.datasets.get_cifar10()


# x is （3,32,32） RGB   32 X 32
# y si a label

# 有5万个训练数据，1万个测试数据
# print('len(train), type ', len(train), type(train))
# print('len(test), type ', len(test), type(test))
#
# print('train[0]', type(train[0]), len(train[0]))
#
# x0, y0 = train[0]
# print('train[0][0]', x0.shape, x0)
# print('train[0][1]', y0.shape, y0, '->', CIFAR10_LABELS_LIST[y0])


def plot_cifar(filepath, data, row, col, scale=3., label_list=None):
    fig_width = data[0][0].shape[1] / 80 * row * scale
    fig_height = data[0][0].shape[2] / 80 * col * scale
    fig, axes = plt.subplots(row,
                             col,
                             figsize=(fig_height, fig_width))
    for i in range(row * col):
        # 画出每张子图
        # train[i][0] is i-th image data with size 32x32
        image, label_index = data[i]
        image = image.transpose(1, 2, 0)
        # i除以col，获得商和余数
        r, c = divmod(i, col)
        # cmap='gray' is for black and white picture.
        axes[r][c].imshow(image)
        if label_list is None:
            axes[r][c].set_title('label {}'.format(label_index))
        else:
            axes[r][c].set_title('{}: {}'.format(label_index, label_list[label_index]))
        axes[r][c].axis('off')
    # automatic padding between subplots
    plt.tight_layout()
    plt.savefig(filepath)
    plt.show()


basedir = 'result/'
# plot_cifar(os.path.join(basedir, 'cifar10_plot.png'), train, 4, 5,
#            scale=4., label_list=CIFAR10_LABELS_LIST)

###########################################################
# CIFAR-100

CIFAR100_LABELS_LIST = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm'
]

train_cifar100, test_cifar100 = chainer.datasets.get_cifar100()


# 训练集5万个数据  测试集1万个数据
# print('len(train_cifar100), type ', len(train_cifar100), type(train_cifar100))
# print('len(test_cifar100), type ', len(test_cifar100), type(test_cifar100))
#
# print('train_cifar100[0]', type(train_cifar100[0]), len(train_cifar100[0]))
#
# x0, y0 = train_cifar100[0]
# print('train_cifar100[0][0]', x0.shape)  # , x0
# print('train_cifar100[0][1]', y0.shape, y0)

# plot_cifar(os.path.join(basedir, 'cifar100_plot_more.png'), train_cifar100,
#            10, 10, scale=4., label_list=CIFAR100_LABELS_LIST)


class CNN(chainer.Chain):
    def __init__(self, class_num):
        super(CNN, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 16, 3, 1)
            self.conv2 = L.Convolution2D(16, 32, 3, 2)
            self.conv3 = L.Convolution2D(32, 32, 3, 1)
            self.conv4 = L.Convolution2D(32, 64, 3, 2)
            self.conv5 = L.Convolution2D(64, 64, 3, 1)
            self.conv6 = L.Convolution2D(64, 128, 3, 2)
            self.fc7 = L.Linear(None, 100)
            self.fc8 = L.Linear(100, class_num)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.relu(self.conv5(h))
        h = F.relu(self.conv6(h))
        h = F.relu(self.fc7(h))
        h = self.fc8(h)
        return h


# 训练数据
archs = {'cnn': CNN}
GPU = 0
class_num = 10
BATCH_SIZE = 16
model = archs['cnn'](class_num=class_num)
EPOCH = 10
OUT = 'result/'

RESUME = 'result/cifar-10.model'

classifier_model = L.Classifier(model)

if GPU >= 0:
    chainer.cuda.get_device(GPU).use()
    classifier_model.to_cpu()

# create a optimizer
optimizer = chainer.optimizers.Adam()
optimizer.setup(classifier_model)

train, test = chainer.datasets.get_cifar10()
train_iter = chainer.iterators.SerialIterator(train, BATCH_SIZE)
test_iter = chainer.iterators.SerialIterator(test, BATCH_SIZE, repeat=False, shuffle=False)

updater = chainer.training.StandardUpdater(train_iter, optimizer, device=GPU)

trainer = chainer.training.Trainer(updater, (EPOCH, 'epoch'), out=OUT)
trainer.extend(extensions.Evaluator(test_iter, classifier_model, device=GPU))
trainer.extend(extensions.snapshot(), trigger=(1, 'epoch'))
trainer.extend(extensions.dump_graph('main/loss'))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(
    ['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                     x_key='epoch', filename='loss.png'))

trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'],
                                     x_key='epoch', filename='accuracy.png'))

trainer.extend(extensions.ProgressBar())

if os.path.exists(RESUME):
    chainer.serializers.load_npz(RESUME, model)

# trainer.run()
# chainer.serializers.save_npz(RESUME, model)


def plot_predict_cifar(filepath, model, data, row, col,
                       scale=3., label_list=None):
    fig_width = data[0][0].shape[1] / 80 * row * scale
    fig_height = data[0][0].shape[2] / 80 * col * scale
    fig, axes = plt.subplots(row,
                             col,
                             figsize=(fig_height, fig_width))
    for i in range(row * col):
        # train[i][0] is i-th image data with size 32x32
        image, label_index = data[i]
        xp = chainer.cuda.cupy
        x = chainer.Variable(xp.asarray(image.reshape(1, 3, 32, 32)))  # test data
        # t = Variable(xp.asarray([test[i][1]])) # labels
        y = model(x)  # Inference result
        prediction = y.data.argmax(axis=1)
        image = image.transpose(1, 2, 0)
        print('Predicted {}-th image, prediction={}, actual={}'
              .format(i, prediction[0], label_index))
        r, c = divmod(i, col)
        axes[r][c].imshow(image)
        if label_list is None:
            axes[r][c].set_title('Predict:{}, Answer: {}'
                                 .format(label_index, prediction[0]))
        else:
            pred = int(prediction[0])
            axes[r][c].set_title('Predict:{} {}\nAnswer:{} {}'
                                 .format(label_index, label_list[label_index],
                                         pred, label_list[pred]))
        axes[r][c].axis('off')
    plt.tight_layout(pad=0.01)
    plt.savefig(filepath)
    plt.show()
    print('Result saved to {}'.format(filepath))


#  保存预测结果
basedir = 'result/'
plot_predict_cifar(os.path.join(basedir, 'cifar10_predict.png'), model,
                   test, 4, 5, scale=5., label_list=CIFAR10_LABELS_LIST)
