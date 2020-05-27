import os
import time

import torch
import torch.optim as optim
from my_models import network
from torch import nn
from torch.autograd import Variable
from torchvision import transforms

import dataset

# 准备数据
DATA_PATH = '/home/straw/Downloads/dataset/kaggle/cifar10/'
BATCH_SIZE = 32
EPOCH = 200
is_pretrain = True
lr = 0.003
momentum = 0.9
# beta1 = 0.5
# beta2 = 0.999
L1_lambda = 100
# train_valid_rate = 0.8
# 保存模型的路径
MODEL_PATH = '/home/straw/Downloads/models/kaggle/cifar10/'
PRETRAIN_MODEL_PATH = os.path.join(MODEL_PATH, 'generator_param_resnet101.pkl')
DATASETS_PATH = "/home/straw/Downloads/dataset/"
# k折交叉验证的计算方法
K = 5
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

train_data, test_data = dataset.get_cifar10(DATASETS_PATH, train_transform=train_transforms,
                                            test_transform=test_transforms)
# import numpy as np
#
# train_data = np.array(train_data)
# n_train_data = int(len(train_data) * train_valid_rate)
# train_data, val_data = torch.utils.data.random_split(train_data, [n_train_data, len(train_data) - n_train_data])
# trainloader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True, num_workers=2)
# validloader = torch.utils.data.DataLoader(val_data, batch_size=16, shuffle=True, num_workers=2)
# testloader = torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=False, num_workers=2)

# ===========================模型准备==================================
net = network()
# 加载预训练的模型
if os.path.exists(PRETRAIN_MODEL_PATH) and is_pretrain:
    net.load_state_dict(torch.load(PRETRAIN_MODEL_PATH))
net.cuda()

# =============================loss=============================
CE_loss = nn.CrossEntropyLoss()

# =========================optimizer===========================
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.8)  # 设置学习率下降策略

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


# k-fold cross validation（k-折叠交叉验证）
# 将n份数据分为n_folds份，以次将第i份作为测试集，其余部分作为训练集
def KFold(num, n_folds=5):
    """
    k折叠交叉验证算法,如果份数不能整除数据的数量
    会尽量分成相近的数量
    :param num: 所有数据的数量
    :param n_folds: k，分成k份
    :return:
    """
    folds = []
    indics = list(range(num))

    for i in range(n_folds):
        # 将一部分数据作为验证集
        valid_indics = indics[(i * num // n_folds):((i + 1) * num // n_folds)]
        # 剩余部分作为测试集
        train_indics = [index for index in range(0, (i * num // n_folds))]
        train_indics.extend([index for index in range((i + 1) * num // n_folds, num)])
        folds.append([train_indics, valid_indics])
    return folds


def train(epoch):
    loss = []
    # 训练开始的时间
    epoch_start_time = time.time()
    net.train()
    num_iter = 0
    correct = 0
    total = 0
    k_index = 0
    for train_indics, valid_indics in KFold(len(train_data), K):
        train_k_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(train_data, train_indics), batch_size=16,
                                                     shuffle=True, num_workers=2)
        valid_k_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(train_data, valid_indics), batch_size=16,
                                                     shuffle=True, num_workers=2)
        for x_, y_ in train_k_loader:
            # put the data into GPU
            x_, y_ = Variable(x_.cuda()), Variable(y_.cuda())
            predict = net(x_)
            # 要优化器梯度清零，而不是网络的梯度清零！！！！
            # 反向传播之前把梯度清为0
            optimizer.zero_grad()

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
                print(
                    "Epoch:{}\t iter:{}\t K:{}\t loss:{:6}\t accuracy:{:6}".format(epoch, num_iter, k_index,
                                                                                   train_loss.data,
                                                                                   (predict == y_).sum().float() / len(
                                                                                       y_)))
        train_accu = (correct / total).cpu().detach().data.numpy()
        # 一波数据训练结束就修改一次学习率
        scheduler.step()
        net.eval()
        with torch.no_grad():
            valid_accu = 0
            valid_loss = 0
            total = 0
            # 使用验证集数据
            for x_, y_ in valid_k_loader:
                x_, y_ = Variable(x_.cuda()), Variable(y_.cuda())
                net.zero_grad()
                predict = net(x_)
                valid_loss += CE_loss(predict, y_) * len(y_)
                valid_accu += accuracy(predict, y_) * len(y_)
                total += len(y_)
            valid_accu /= total
            valid_loss /= total
            # 必须使用loss的data属性，不能直接print cuda上的数据
            print("Epoch{}:\t K{}:\t valid_loss:{:.6f}\t valid_accu:{:.6f}".format(epoch, k_index, valid_loss,
                                                                                   valid_accu))

        k_index += 1

    per_epoch_time = time.time() - epoch_start_time
    print('[{}/{}] - use {:.2f} minutes\t loss: {:.6f}\t accu: {:.6f}\t'.format(
        (epoch + 1), EPOCH, per_epoch_time / 60, torch.mean(torch.FloatTensor(loss)),
        torch.mean(torch.FloatTensor(train_accu))))
    # k折训练结束之后保存模型
    train_hist['per_epoch_ptimes'].append(per_epoch_time)
    print('EPOCH {} is over! ... save training results'.format(epoch))
    import time_util
    torch.save(net.state_dict(), os.path.join(MODEL_PATH, 'res101_ep{}_{}.pkl'.format(epoch, time_util.get_date())))


# 训练数据
for epoch in range(EPOCH):
    train(epoch)
