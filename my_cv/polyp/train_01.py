import os

import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torchvision import transforms

import file_util
import time_util
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
# 训练集数据总共的训练集数据中的个数
VALID_RATE = 0.2

# 正式开始测试，将数据分为训练集，测试机和验证机
# 186
# 120用于训练
# 30张数据用于验证
# 36用于测试

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

# 准备数据
train_data = dataset(DATA_PATH, MASK_PATH, image_transforms, mask_transforms)
# train_data = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
# train_data = np.array(train_data)
n_train_data = int(len(train_data) * VALID_RATE)
train_data, val_data = torch.utils.data.random_split(train_data, [n_train_data, len(train_data) - n_train_data])
train_loader = torch.utils.data.DataLoader(train_data)
val_laoder = torch.utils.data.DataLoader(val_data)

# 设置输出通道为1
net = FCN(n_out=1)
if is_use_gpu:
    net = net.cuda()
if is_pretrain:
    net.load_state_dict(torch.load(PRETRAIN_PATH))

optimizer = optim.SGD(net.parameters(), lr=lr)
# 创建loss函数
if is_use_gpu:
    # loss_fucntion = CrossEntropyLoss().cuda()
    # loss_fucntion = NLLLoss().cuda()
    loss_fucntion = nn.BCEWithLogitsLoss().cuda()
else:
    # loss_fucntion = NLLLoss()
    loss_fucntion = nn.BCEWithLogitsLoss()

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
    last_out = None
    last_mask = None
    last_image = None
    # 开始每个批次
    for image, mask in train_data:
        # 检查输入的图像是否有问题
        # from PIL import Image
        #
        # image = image.numpy()
        # image = image[0, :, :, :]
        # print(image.size)
        # Image.fromarray(image).show()

        num_iter += 1
        if is_use_gpu:
            # 将数据放入GPU
            image, mask = Variable(image.cuda()), Variable(mask.cuda())
            # label = Variable(label.cuda())
            # label = Variable(label.long().cuda())
        else:
            image, mask = Variable(image), Variable(mask)
            # label = Variable(label.long())

        optimizer.zero_grad()
        # 将数据输入网络
        out = net(image)
        # 设置softmax，将结果的范围固定在0和1之间
        # out = F.log_softmax(out, dim=1)
        # 计算Loss
        # todo 损失函数需要改进
        # print(out.size())
        # 不把多余的通道数1，去掉的话会报错
        # label = torch.squeeze(label)
        # print(label.size())
        # if last_out is not None and torch.equal(out, last_out):
        #     print("out")
        # if last_mask is not None and torch.equal(label, last_mask):
        #     print("mask")
        # if last_image is not None and torch.equal(image, last_image):
        #     print("image")
        loss = loss_fucntion(out, mask)
        # print(loss.data.size())
        # 返回数据
        loss.backward()
        optimizer.step()
        # 计算累计的损失
        train_loss = loss.data + train_loss
        print(train_loss)
    train_loss = train_loss / num_iter
    print("EPOCH:{} train_loss:{}".format(epoch, train_loss))

    print("==============saving model data===============")

    torch.save(net.state_dict(),
               os.path.join(MODEL_PATH, 'FCN_NLL_ep{}_{}_second.pkl'.format(epoch, time_util.get_date())))
    print("==============saving is over===============")

print("================training is over=================")
