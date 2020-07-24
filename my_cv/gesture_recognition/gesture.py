import chainer.functions as F
import chainer.links as L
import chainer
from PIL import Image
import numpy as np
from chainer import reporter
from chainer import training
from chainer.training import extensions
from chainer import serializers
from chainer.links import VGG16Layers
from chainer import Variable
import os


# 建立模型
class HandModel(chainer.Chain):
    def __init__(self):
        super(HandModel, self).__init__()
        with self.init_scope():
            # self.l1 = L.Linear(None, 1024)
            # self.l2 = L.Linear(None, 512)
            # self.l3 = L.Linear(None, 128)
            # self.l4 = L.Linear(None, 32)
            # model =
            #    feature's shape is (1,4096)
            # feature = model.extract([x], layers=["fc7"])["fc7"]
            # feature =
            self.feature_extract = VGG16Layers()
            # self.feature_extract.train = False
            self.conv1 = L.Convolution2D(
                in_channels=128, out_channels=64, ksize=5, stride=1, pad=2)
            self.conv2 = L.Convolution2D(
                in_channels=64, out_channels=32, ksize=5, stride=1, pad=2)
            self.conv3 = L.Convolution2D(
                in_channels=32, out_channels=16, ksize=5, stride=1, pad=2)
            # self.fc4 = L.Linear(None, 84)
            # self.fc5 = L.Linear(84, 10)

    def __call__(self, *args, finetune=False):
        output = self.forward(*args)
        truth = args[1]
        # todo calcu loss
        self.loss = self.loss_funtion(output, truth)
        # self.loss = F.mean_squared_error(output, truth)
        reporter.report({'loss': self.loss}, self)
        # print(self.loss)
        return self.loss

    def loss_funtion(self, output, truth):
        loss = []
        batch_size = len(output)
        for i in range(batch_size):
            my_mse_loss = 0
            for o, t in zip(output[:, i], truth[:, i]):
                # todo 应该有使用int以外的更好的办法
                # temp = int(t.argmax())
                # matrix = ((o - t) ** 2)
                # my_mse_loss = F.sum(matrix) + matrix[temp // 56, temp % 56] * 100000
                # my_mse_loss = matrix[temp // 56, temp % 56] * 10000
                # my_mse_loss *= (1.0 / (56 * 56))
                # my_mse_loss = F.mean_squared_error(o, t)
                my_mse_loss = F.mean_absolute_error(o, t)
                # print(t[temp // 56, temp % 56])
                # print(my_mse_loss)
                # matrix[temp // 56, temp % 56] = 1000 * matrix[temp // 56, temp % 56]
                my_mse_loss += my_mse_loss
                # print(my_mse_loss)
                # loss.append(matrix)
            loss.append(my_mse_loss)
        return sum(loss) / batch_size

    def forward(self, *args, finetune=False):
        x = args[0]
        # x = Variable(x)
        feature = self.feature_extract(x, layers=['pool2'])["pool2"]
        feature = Variable(feature.data)
        feature.volatile = True
        # todo
        # feature = Variable(feature.data, volatile=True)
        # feature = model.extract([x], layers=["fc7"])["fc7"]
        # feature = model.extract([x], layers=["pool2"])["pool2"]
        # feature = model.extract([x], layers=["pool5"])["pool5"]
        # h = F.relu(self.l1(feature))
        # h = F.relu(self.l2(h))
        # h = F.relu(self.l3(h))
        h = F.relu(self.conv1(feature))
        h = F.relu(self.conv2(h))
        # 16 * 56 * 56
        return F.sigmoid(self.conv3(h))
        # h = F.relu(self.conv3(h))
        # return self.l4(h)

    def predict(self, *args):
        with chainer.using_config('train', False):
            with chainer.no_backprop_mode():
                return self.forward(*args)

    # def predict2(self, *args, batchsize=8):
    #     data = args[0]
    #     x_list = []
    #     y_list = []
    #     t_list = []
    #     for i in range(0, len(data), batchsize):
    #         x, t = concat_examples(data[i:i + batchsize])
    #         y = self.predict(x)
    #         y_list.append(y.data)
    #         x_list.append(x)
    #         t_list.append(t)
    #
    #     x_array = np.concatenate(x_list)[:, 0]
    #     y_array = np.concatenate(y_list)[:, 0]
    #     t_array = np.concatenate(t_list)[:, 0]
    #     return x_array, y_array, t_array


# 建立数据读取器
class HandDataSet(chainer.dataset.DatasetMixin):
    # 也许可以用PickleDataset来改进
    def __init__(self, image_path, label_path):
        self.images_name = []
        for file in os.listdir(image_path):
            # full_path = os.path.join(image_path, file)
            self.images_name.append(file)
            # label_full_path
        self.image_path = image_path
        self.label_path = label_path
        # self.base = chainer.datasets.ImageDataset(path_file)

    def __len__(self):
        return len(self.images_name)

    def get_example(self, index):
        image_name = self.images_name[index]
        image_path = os.path.join(self.image_path, image_name)
        label_path = os.path.join(self.label_path, image_name.replace('jpg', 'npy'))
        image = np.array(Image.open(image_path), dtype=np.float32)
        label = np.load(label_path)
        image *= (1.0 / 255.0)
        image = image.transpose(2, 0, 1)

        # _, h, w = image.shape
        # TODO maybe need some transforms
        # height, width, _ = image.shape
        # heat_map = np.zeros([16, height, width])
        heat_map = np.zeros([16, 56, 56], dtype=np.float32)

        for i, point in enumerate(label):
            heat_map[i, point[0] // 4, point[1] // 4] = 1
        return image, heat_map


# 准备数据
image_path = '/home/straw/PycharmProjects/straw/my_cv/hand_gesture/result/images'
label_path = 'result/'
SAVE_PATH = 'result/'
hand_dataset = HandDataSet(image_path, label_path)
BATCH_SIZE = 4
EPOCH = 10
GPU = 0
train_ratio = 0.7
train_size = int(len(hand_dataset) * train_ratio)
train, test = chainer.datasets.split_dataset_random(hand_dataset, train_size, seed=1)
train_iter = chainer.iterators.SerialIterator(train, BATCH_SIZE)
test_iter = chainer.iterators.SerialIterator(train, BATCH_SIZE, repeat=False, shuffle=False)

# 建立模型
model = HandModel()
# 建立优化器
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)

# 创建训练器
updater = training.StandardUpdater(train_iter, optimizer, device=GPU)
trainer = training.Trainer(updater, (EPOCH, 'epoch'), out=SAVE_PATH)
trainer.extend(extensions.Evaluator(test_iter, model, device=GPU))
trainer.extend(extensions.dump_graph('main/loss'))
trainer.extend(extensions.snapshot(), trigger=(1, 'epoch'))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(
    ['epoch', 'main/loss', 'validation/main/loss', 'elapsed_time']))
# Plot graph for loss for each epoch
if extensions.PlotReport.available():
    trainer.extend(extensions.PlotReport(
        ['main/loss', 'validation/main/loss'],
        x_key='epoch', file_name='loss.png'))
else:
    print('Warning: PlotReport is not available in your environment')
trainer.extend(extensions.ProgressBar())
# load trained data,if have
RESUME = 'result/hand.model'
# RESUME = None

if RESUME:
    # Resume from a snapshot
    serializers.load_npz(RESUME, model)

# 训练
trainer.run()
# 保存
serializers.save_npz('{}/hand.model'.format(SAVE_PATH), model)
# 测试
