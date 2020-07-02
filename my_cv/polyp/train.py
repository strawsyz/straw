import os

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

import file_util
import time_util
from dataset import PolypDataset as dataset
from models import FCN

# 将vgg改为ResNet

# EPOCH:1989 train_loss:0.287866
# Epoch1989:	 valid_loss:0.328060
# ==============saving model data===============
# ==============saving at /home/straw/Downloads/models/polyp/2020-06-26/FCN_NLL_ep1989_04-57-10.pkl===============

# 设置训练参数
# 由于最后处理的时候要将去掉通道数1的通道，所以不能设置为1

image_transforms = transforms.Compose([
    # 随机调整亮度，对比度，饱和度，色相
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    # 随机水平翻转
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomVerticalFlip(),
    # transforms.RandomRotation(100),

    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # transforms.Normalize(mean=[0.482, 0.456, 0.406],
    #                      std=[0.229, 0.224, 0.225])
])
mask_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    # 将输出设置为一个通道
    transforms.Grayscale(),
    transforms.ToTensor(),
])


def prepare_data():
    # 为了使结果可以复现，确性随机分的数据是什么
    torch.manual_seed(7)  # cpu
    torch.cuda.manual_seed(7)  # gpu
    import numpy as np
    import random
    np.random.seed(7)  # numpy
    random.seed(7)  # random and transforms
    torch.backends.cudnn.deterministic = True  # cudnn

    # 准备数据
    train_data = dataset(DATA_PATH, MASK_PATH, image_transforms, mask_transforms)
    # 设置全部训练数据集的大小
    train_data.set_data_num(N_TRAIN)
    # train_data = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    # train_data = np.array(train_data)
    n_train_data = int(N_TRAIN * VALID_RATE)
    train_data, val_data = torch.utils.data.random_split(train_data, [n_train_data, N_TRAIN - n_train_data])
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)
    return train_loader, val_loader


def prepare_net(optim="adam"):
    # 设置输出通道为1
    net = FCN(n_out=1)
    loss_function = nn.BCEWithLogitsLoss()

    if is_use_gpu:
        net = net.cuda()
        loss_function = loss_function.cuda()
    if is_pretrain:
        print("load the model from {}".format(PRETRAIN_PATH))
        load_checkpoint(PRETRAIN_PATH)
        # net.load_state_dict(torch.load(PRETRAIN_PATH))

    if optim == "adam":
        optimizer = optim.Adam(net.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.8)  # 设置学习率下降策略

    # 创建loss函数
    # if is_use_gpu:
    # loss_fucntion = CrossEntropyLoss().cuda()
    # loss_fucntion = NLLLoss().cuda()
    # else:
    # loss_fucntion = NLLLoss()
    return net, optimizer, loss_function, scheduler


def train(epoch, optim):
    net.train()
    train_loss = 0
    n_total = 0
    for image, mask in train_loader:
        n_total += len(image)
        if is_use_gpu:
            # 将数据放入GPU
            image, mask = Variable(image.cuda()), Variable(mask.cuda())
        else:
            image, mask = Variable(image), Variable(mask)

        optimizer.zero_grad()
        # 将数据输入网络
        out = net(image)
        # 设置softmax，将结果的范围固定在0和1之间
        # out = F.log_softmax(out, dim=1)
        # 计算Loss
        # todo 损失函数需要改进
        loss = loss_function(out, mask)
        # 返回数据
        loss.backward()
        optimizer.step()
        # 计算累计的损失
        train_loss = loss.data + train_loss
    train_loss = train_loss / n_total
    print("EPOCH:{} train_loss:{:.6f}".format(epoch, train_loss))

    net.eval()
    with torch.no_grad():
        valid_loss = 0
        total = 0
        # 使用验证集数据
        for image, mask in val_loader:
            image, mask = Variable(image.cuda()), Variable(mask.cuda())
            net.zero_grad()
            predict = net(image)
            valid_loss += loss_function(predict, mask)
            total += len(image)
        valid_loss /= total
        # 必须使用loss的data属性，不能直接print cuda上的数据
        print("Epoch{}:\t valid_loss:{:.6f}\t ".format(epoch, valid_loss))
    scheduler.step()


import shutil


def save_checkpoint(is_best):
    print("==============saving model data===============")
    save_path = os.path.join(MODEL_PATH,
                             'FCN_NLL_ep{}_{}.pkl'.format(epoch, time_util.get_time("%H-%M-%S")))
    state = {
        'epoch': epoch + 1,
        'state_dict': net.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, save_path)
    if is_best:
        shutil.copyfile(save_path, 'model_best.pth.tar')

    print("==============saving at {}===============".format(save_path))


def load_checkpoint(path):
    if os.path.isfile(path):
        print("=" * 10 + " loading checkpoint '{}'".format(path) + "=" * 10)
        checkpoint = torch.load(path)
        start_epoch.load_state_dict(checkpoint['epoch'])
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(path, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(path))


if __name__ == '__main__':
    BATCH_SIZE = 2
    EPOCH = 500
    is_use_gpu = True
    DATA_PATH = "/home/straw/Downloads/dataset/polyp/TMP/05/data"
    MASK_PATH = "/home/straw/Downloads/dataset/polyp/TMP/05/mask"
    MODEL_PATH = "/home/straw/Downloads/models/polyp/"
    MODEL_PATH = os.path.join(MODEL_PATH, time_util.get_date())
    file_util.make_directory(MODEL_PATH)
    lr = 0.002
    # is_pretrain = True
    is_pretrain = False

    # 用于训练和验证的所有数据集
    N_TRAIN = 600
    # 训练集数据总共的训练集数据中的百分比
    VALID_RATE = 0.2
    PRETRAIN_PATH = ""

    # 准备数据
    train_loader, val_loader = prepare_data()
    optim = "adam"
    net, optimizer, loss_function, scheduler = prepare_net(optim=optim)
    eval_miou = []
    best = [0]
    start_epoch = 0
    print("================training start=================")
    for epoch in range(start_epoch, start_epoch + EPOCH):
        train(epoch)
        # todo 增加是否保存模型的判断
        save_checkpoint(is_best=False)
    print("================training is over=================")
