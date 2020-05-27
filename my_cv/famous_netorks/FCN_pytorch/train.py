# -*- coding: utf-8 -*-

from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataset import NYU
from models import FCN
from torch.nn import NLLLoss

# 设置训练参数
BATCH_SIZE = 2
EPOCH = 500
# 学习率
lr = 1e-4

TRAIN_DATA_PATH = ""
TRAIN_LABEL_PATH = ""
VAL_DATA_PATH = ""
VAL_LABEL_PATH = ""

# 创建数据加载器
train_data = DataLoader(NYU(TRAIN_DATA_PATH, TRAIN_LABEL_PATH), batch_size=BATCH_SIZE,
                        shuffle=True, num_workers=4, phase_train=True)
val_data = DataLoader(NYU(VAL_DATA_PATH, VAL_LABEL_PATH), batch_size=BATCH_SIZE,
                      shuffle=True, num_workers=4, phase_train=False)

net = FCN().cuda()
optimizer = optim.Adam(net.parameters(), lr=lr)

eval_miou = []
best = [0]

print("===============start training================")

for epoch in range(EPOCH):
    train_loss = 0
    # train_acc = 0
    # train_miou = 0
    # train_class_acc = 0
    # 设置网络为训练模式
    net = net.train()
    num_iter = 0
    # 开始每个批次
    for sample in train_data:
        num_iter += 1
        # 将数据放入GPU
        image = Variable(sample['image'].cuda())
        label = Variable(sample['label'].long().cuda())

        optimizer.zero_grad()
        # 将数据输入网络
        out = net(image)
        # 设置softmax
        out = F.log_softmax(out, dim=1)
        # 计算Loss
        loss = NLLLoss(out, label)
        # 返回数据
        loss.backward()
        optimizer.step()
        # 计算累计的损失
        train_loss = loss.item() + train_loss

    train_loss = train_loss / num_iter
    print("EPOCH:{}".format(epoch))
    print("train_loss:{}".format(train_loss))

print("================训练结束=================")
