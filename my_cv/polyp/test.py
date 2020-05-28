# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

import file_util
from dataset import PolypDataset as dataset
from models import FCN

# 设置训练参数
# 由于最后处理的时候要将去掉通道数1的通道，所以不能设置为1
BATCH_SIZE = 32
is_use_gpu = True
DATA_PATH = "/home/straw/Downloads/dataset/polyp/data/"
MASK_PATH = "/home/straw/Downloads/dataset/polyp/mask/"
# 使用的预训练模型的路径
PRETRAIN_PATH = "/home/straw/Downloads/models/polyp/FCN_NLL_ep413_2020-05-25.pkl"
RESULT_SAVE_PATH = "/home/straw/Downloads/dataset/polyp/result/"

image_transforms = transforms.Compose([
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

# 准备数据
test_data = DataLoader(dataset(DATA_PATH, MASK_PATH, image_transforms, mask_transforms), batch_size=BATCH_SIZE,
                       shuffle=True)
# 设置输出通道为1
net = FCN(n_out=1)
if is_use_gpu:
    net = net.cuda()

net.load_state_dict(torch.load(PRETRAIN_PATH))

# 创建loss函数
if is_use_gpu:
    loss_fucntion = nn.BCEWithLogitsLoss().cuda()
else:
    loss_fucntion = nn.BCEWithLogitsLoss()
# optimizer = optim.SGD(net.parameters(), lr=lr)

# 变成验证模式
net.eval()
import os
import time_util

# 当前的设置当时会按照天数来保存文件
# todo 有必要改进
RESULT_SAVE_PATH = os.path.join(RESULT_SAVE_PATH, time_util.get_date())
file_util.make_directory(RESULT_SAVE_PATH)
print("=" * 10 + "test start" + "=" * 10)
for i, (image, mask, image_name) in enumerate(test_data):
    # 将数据放入GPU
    if is_use_gpu:
        image, mask = Variable(image).cuda(), Variable(mask).cuda()
    else:
        image, mask = Variable(image), Variable(mask)
    # optimizer.zero_grad()
    out = net(image)

    loss = loss_fucntion(image, mask)
    predict = out.max(1)[1].squeeze().cpu().data().numpy()
    predict = Image.fromarray(predict)
    # 默认应该全都保存为png文件
    predict.save(os.path.join(RESULT_SAVE_PATH, image_name))

print("================testing end=================")
