# -*- coding: utf-8 -*-
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataset import NYU
from models import FCN
from PIL import Image

# 设置训练参数
# 由于最后处理的时候要将去掉通道数1的通道，所以不能设置为1
BATCH_SIZE = 1
EPOCH = 500
# 学习率
lr = 0.0001
is_use_gpu = True
DATA_PATH = "/home/straw/Downloads/dataset/polyp/data/"
MASK_PATH = "/home/straw/Downloads/dataset/polyp/mask/"
MODEL_PATH = "/home/straw/Downloads/models/polyp/"
PRETRAIN_PATH = "/home/straw/Downloads/models/polyp/FCN_NLL_ep413_2020-05-25.pkl"
is_pretrain = True

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
train_data = DataLoader(dataset(DATA_PATH, MASK_PATH, image_transforms, mask_transforms), batch_size=BATCH_SIZE,
                        shuffle=True)
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



# 变成验证模式
net.eval()
# 加载模型
net.load_state_dict(net.load(MODEL_PATH))

print("===============start testing================")
for i, (data, label) in enumerate(test_data):
    # 将数据放入GPU
    data = Variable(data).cuda()
    label = Variable(label).cuda()
    # 用预测结果
    out = F.log_softmax(net(data), dim=1)
    predict = out.max(1)[1].squeeze().cpu().data().numpy()
    predict = Image.fromarray(predict)
    # 结果
    predict.save("{}{}.png".format(SAVE_PATH, str(i)))

print("================testing end=================")
