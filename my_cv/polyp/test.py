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
BATCH_SIZE = 16  # 设置为32 ，内存就炸了
N_TEST = 140
is_use_gpu = True
DATA_PATH = "/home/straw/Downloads/dataset/polyp/data/"
MASK_PATH = "/home/straw/Downloads/dataset/polyp/mask/"

# 使用的预训练模型的路径
PRETRAIN_PATH = "/home/straw/Downloads/models/polyp/2020-06-04/FCN_NLL_ep156_12-39-27.pkl"
PRETRAIN_PATH = "/home/straw/Downloads/models/polyp/2020-06-05/FCN_NLL_ep91_15-27-52.pkl"

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


def prepare_data():
    # 准备数据
    tset_data = dataset(DATA_PATH, MASK_PATH, image_transforms, mask_transforms, test=True)
    # 设置全部训练数据集的大小
    tset_data.set_data_num(N_TEST)
    # train_data = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    # train_data = np.array(train_data)
    # n_train_data = int(N_TRAIN * VALID_RATE)
    # train_data, val_data = torch.utils.data.random_split(train_data, [n_train_data, N_TRAIN - n_train_data])
    test_loader = DataLoader(tset_data, batch_size=BATCH_SIZE, shuffle=True)
    return test_loader




# polpy_dataset = dataset(DATA_PATH, MASK_PATH, image_transforms, mask_transforms, test=True)
# polpy_dataset.set_data_num(4)

def prepare_net():
    net = FCN(n_out=1)
    if is_use_gpu:
        net = net.cuda()

    # 加载预训练模型
    if not os.path.exists(PRETRAIN_PATH):
        print("Can't find the pretrain model")
    net.load_state_dict(torch.load(PRETRAIN_PATH))

    # 创建loss函数
    if is_use_gpu:
        loss_fucntion = nn.BCEWithLogitsLoss().cuda()
    else:
        loss_fucntion = nn.BCEWithLogitsLoss()
    # optimizer = optim.SGD(net.parameters(), lr=lr)
    return net, loss_fucntion


test_loader = prepare_data()

net, loss_function = prepare_net()

# 变成验证模式
net.eval()
import os
import time_util

# 当前的设置当时会按照天数来保存文件
# todo 有必要改进
RESULT_SAVE_PATH = os.path.join(RESULT_SAVE_PATH, time_util.get_date())
file_util.make_directory(RESULT_SAVE_PATH)
print("=" * 10 + "test start" + "=" * 10)
sigmoid = nn.Sigmoid()
for i, (image, mask, image_name) in enumerate(test_loader):
    # 将数据放入GPU
    if is_use_gpu:
        image, mask = Variable(image).cuda(), Variable(mask).cuda()
    else:
        image, mask = Variable(image), Variable(mask)
    # optimizer.zero_grad()
    out = net(image)
    _, _, width, height = mask.size()
    # 计算损失
    loss = loss_function(out, mask)
    print(loss)

    predict = out.squeeze().cpu().data.numpy()
    # 画出图像
    # predict = out.max(1)[1].squeeze().cpu().data.numpy()
    # if (predict == 0).all():
    #     print(1)
    # 数据分析
    # for tmp in predict:
    #     for i in tmp:
    #         ser = pd.Series(i)
    #         print(ser.mode())

    for index, pred in enumerate(predict):
        # 训练模型
        # pred = sigmoid(torch.from_numpy(pred.astype(np.float)))
        # pred = pred.data.numpy()
        pred = pred * 255
        pred = pred.astype('uint8')
        # print(pred)
        pred = Image.fromarray(pred)
        # pred.convert('L')
        # 恢复到原图像的大小
        pred = pred.resize((width, height))
        save_path = os.path.join(RESULT_SAVE_PATH, image_name[index])
        # pred.show()
        # 默认应该全都保存为png文件
        pred.save(save_path)
        print("================{}=================".format(save_path))

print("================testing end=================")
