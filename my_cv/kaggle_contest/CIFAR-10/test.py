import os

import pandas as pd
import torch
from torch.autograd import Variable
from torchvision import transforms

import dataset
from my_models import network

MODEL_PATH = '/home/straw/Downloads/models/kaggle/cifar10/generator_param_resnet101.pkl'
RESULT_PATH = "/home/straw/Downloads/dataset/kaggle/cifar10/sub.csv"
# 标签列表
LABEL_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']



def get_label_name(index):
    return [LABEL_NAMES[i] for i in index]


def test():
    # 创建网络
    net = network()
    # 加载之前保存的参数
    net.load_state_dict(torch.load())
    net.cuda()
    net.eval()
    result_labels = []
    with torch.no_grad():
        # 使用验证集数据
        for i, x_ in enumerate(testloader):
            x_ = Variable(x_.cuda())
            net.zero_grad()
            predict = net(x_)
            predict = predict.argmax(1)
            # print(predict)
            label_name = get_label_name(predict)
            result_labels.extend(label_name)
            print("======{}=====".format(i))

    return result_labels


def save(result_labels):
    # 保存数据
    df = pd.DataFrame(result_labels, columns=["label"], index=list(range(1, len(result_labels) + 1)))
    df.index.name = 'id'
    df.to_csv(RESULT_PATH)


if __name__ == '__main__':
    # 数据预处理
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    # 准备数据
    test_data = dataset.Cifar10TestDataset(transform=test_transforms)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=256, shuffle=False, num_workers=2)

    result_labels = test()
    save(result_labels)
