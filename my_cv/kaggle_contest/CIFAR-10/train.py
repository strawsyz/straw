import os
import time

import torch
import torch.optim as optim
from torch import nn
from torch.autograd import Variable
from torchvision import transforms

import dataset
import my_models

# 准备数据
DATA_PATH = '/home/straw/下载/dataset/kaggle/cifar10/'
BATCH_SIZE = 32
EPOCH = 200
lr = 0.02
beta1 = 0.5
beta2 = 0.999
# 训练集和验证集比例
train_valid_rate = 0.8
# results save path
MODEL_PATH = '/home/straw/下载/models/kaggle/cifar10/'
# 保存模型的路径
MODEL_PATH = './'

# ===========================数据集准备==================================
# 数据增强的方法
train_transforms = transforms.Compose([
    # 随机调整亮度，对比度，饱和度，色相
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    # 随机水平翻转
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# 加载标签数据，使用字典存储，key是id，value是标签
# labels = util.read_label_file(DATA_PATH, "trainLabels.csv")

datasets_path = "/home/straw/下载/dataset/"
train_data, test_data = dataset.get_cifar10(datasets_path, train_transform=train_transforms,
                                            test_transform=test_transforms)
# print(train_data[0][0].size)
n_train_data = int(len(train_data) * train_valid_rate)
train_data, val_data = torch.utils.data.random_split(train_data, [n_train_data, len(train_data) - n_train_data])
# print(len(train_data))  # 48000
# print(len(val_data))  # 12000

trainloader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True, num_workers=2)
validloader = torch.utils.data.DataLoader(val_data, batch_size=16, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=False, num_workers=2)
# 测试集1万枚数据
# print(len(test_data))

# ===========================模型准备==================================
net = my_models.MobileNetV2()
net.cuda()
# =============================loss=============================
CE_loss = nn.CrossEntropyLoss()
# =========================optimizer===========================
optimizer = optim.Adam(net.parameters(), lr=lr, betas=(beta1, beta2))

train_hist = {}
train_hist['loss'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []

# training the network
print('training start!')
start_time = time.time()


def accuracy(output, labels):
    """计算正确度"""
    # 找到预测的结果
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def train(epoch):
    loss = []
    # 训练开始的时间
    epoch_start_time = time.time()
    net.train()
    num_iter = 0
    correct = 0
    total = 0
    for x_, y_ in trainloader:
        # put the data into GPU
        x_, y_ = Variable(x_.cuda()), Variable(y_.cuda())

        net.zero_grad()

        predict = net(x_)
        train_loss = CE_loss(predict, y_)
        train_loss.backward()
        optimizer.step()

        train_hist['loss'].append(train_loss.data)
        loss.append(train_loss.data)
        num_iter += 1
        predict = torch.argmax(predict, 1)
        # 进行除法之前需要先转换成浮点数
        # print( (predict == y_).sum().float()/len(y_))
        # 计算正确度
        correct += (predict == y_).sum().float()
        total += len(y_)
        if num_iter % 200 == 0:
            print("Epoch:{}\t iter:{}\t loss:{}\t accuracy:{}".format(epoch, num_iter, train_loss.data,
                                                                      (predict == y_).sum().float() / len(y_)))
    acc_str = 'Accuracy: %f' % ((correct / total).cpu().detach().data.numpy())
    print(acc_str)
    net.eval()
    with torch.no_grad():
        valid_accu = 0
        valid_loss = 0
        total = 0
        for x_, y_ in validloader:
            x_, y_ = Variable(x_.cuda()), Variable(y_.cuda())
            net.zero_grad()
            predict = net(x_)
            valid_loss += CE_loss(predict, y_) * len(y_)
            valid_accu += accuracy(predict, y_) * len(y_)
            total += len(y_)
        valid_accu /= total
        valid_loss /= total
        # 必须使用loss的data属性，不能直接print cuda上的数据
        print("Epoch{}:\t valid_loss:{}\t accuracy:{}".format(epoch, valid_loss, valid_accu))

    per_epoch_time = time.time() - epoch_start_time
    print('[%d/%d] - ptime: %.2f, loss: %.3f' % (
        (epoch + 1), EPOCH, per_epoch_time, torch.mean(torch.FloatTensor(loss))))

    train_hist['per_epoch_ptimes'].append(per_epoch_time)
    print('EPOCH {} is over! ... save training results'.format(epoch))
    torch.save(net.state_dict(), os.path.join(MODEL_PATH, 'generator_param.pkl'))


if __name__ == '__main__':
    # 训练数据
    for epoch in range(EPOCH):
        train(epoch)

    print("training is over, use time : {:.2}".format((time.time() - start_time) / 3600))
