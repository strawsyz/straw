import os

import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

import file_util
from dataset import PolypDataset as dataset
from models import FCN

# 设置训练参数
# 由于最后处理的时候要将去掉通道数1的通道，所以不能设置为1
BATCH_SIZE = 1
EPOCH = 500
# 学习率
lr = 0.0003
is_use_gpu = True
DATA_PATH = "/home/straw/Downloads/dataset/polyp/data/"
MASK_PATH = "/home/straw/Downloads/dataset/polyp/mask/"
MODEL_PATH = "/home/straw/Downloads/models/polyp/"
PRETRAIN_PATH = "/home/straw/Downloads/models/polyp/FCN_NLL_ep413_2020-05-25.pkl"
is_pretrain = True
# 用于训练和验证的所有数据集
N_TRAIN = 120
# 训练集数据总共的训练集数据中的百分比
VALID_RATE = 0.2

# 正式开始测试，将数据分为训练集，测试机和验证机
# 186
# 120用于训练
# 30张数据用于验证
# 36用于测试
import time_util

MODEL_PATH = os.path.join(MODEL_PATH, time_util.get_date())
file_util.make_directory(MODEL_PATH)
image_transforms = transforms.Compose([
    # 随机调整亮度，对比度，饱和度，色相
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    # 随机水平翻转
    # transforms.RandomHorizontalFlip(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
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
    # 准备数据
    train_data = dataset(DATA_PATH, MASK_PATH, image_transforms, mask_transforms)
    train_data.set_data_num(N_TRAIN)
    # train_data = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    # train_data = np.array(train_data)
    n_train_data = int(N_TRAIN * VALID_RATE)
    train_data, val_data = torch.utils.data.random_split(train_data, [n_train_data, N_TRAIN - n_train_data])
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)
    return train_loader, val_loader


def prepare_net():
    # 设置输出通道为1
    net = FCN(n_out=1)
    loss_function = nn.BCEWithLogitsLoss()

    if is_use_gpu:
        net = net.cuda()
        loss_function = loss_function.cuda()
    if is_pretrain:
        net.load_state_dict(torch.load(PRETRAIN_PATH))
    optimizer = optim.SGD(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.8)  # 设置学习率下降策略

    # 创建loss函数
    # if is_use_gpu:
    # loss_fucntion = CrossEntropyLoss().cuda()
    # loss_fucntion = NLLLoss().cuda()
    # else:
    # loss_fucntion = NLLLoss()
    return net, optimizer, loss_function, scheduler


train_loader, val_loader = prepare_data()

net, optimizer, loss_function, scheduler = prepare_net()
eval_miou = []
best = [0]


def train(epoch):
    # for epoch in range(EPOCH):
    # 设置网络为训练模式
    net.train()
    num_iter = 0
    train_loss = 0
    n_total = 0
    # 开始每个批次
    for image, mask in train_loader:
        # 检查输入的图像是否有问题
        # num_iter += 1
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
        # print(loss.data.size())
        # 返回数据
        loss.backward()
        optimizer.step()
        # 计算累计的损失
        train_loss = loss.data + train_loss
        print(train_loss)
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
    # todo 增加是否保存模型的判断
    print("==============saving model data===============")
    model_save_path = os.path.join(MODEL_PATH,
                                   'FCN_NLL_ep{}_{}.pkl'.format(epoch, time_util.get_time("%H-%M-%S")))
    torch.save(net.state_dict(), model_save_path)
    print("==============saving at {}===============".format(model_save_path))

    scheduler.step()


if __name__ == '__main__':
    print("================training start=================")
    for epoch in range(EPOCH):
        train(epoch)
    print("================training is over=================")
