from torchvision import datasets

datasets_path = "/home/straw/下载/dataset/"


def get_cifar10(path, train_transform, test_transform):
    train = datasets.CIFAR10(root=path, train=True, download=True, transform=train_transform)
    test = datasets.CIFAR10(root=path, train=False, download=True, transform=test_transform)
    return train, test
import torch
trainset = datasets.CIFAR10(root=datasets_path, train=True,
                                        download=True)
    # trainloader其实是一个比较重要的东西，我们后面就是通过trainloader把数据传入网
    # 络，当然这里的trainloader其实是个变量名，可以随便取，重点是他是由后面的
    # torch.utils.data.DataLoader()定义的，这个东西来源于torch.utils.data模块，
    #  网页链接http://pytorch.org/docs/0.3.0/data.html，这个类可见我后面图2
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)


#
# import os
# import random
# from PIL import Image
# from torch.utils.data import Dataset
#
# random.seed(1)
# rmb_label = {"1": 0, "100": 1}
#
#
# class CifarDataset(Dataset):
#     def __init__(self, data_dir, transform=None):
#         """
#         rmb面额分类任务的Dataset
#         :param data_dir: str, 数据集所在路径
#         :param transform: torch.transform，数据预处理
#         """
#         self.label_name = {"1": 0, "100": 1}
#         self.data_info = self.get_img_info(data_dir)  # data_info存储所有图片路径和标签，在DataLoader中通过index读取样本
#         self.transform = transform
#
#     def __getitem__(self, index):
#         path_img, label = self.data_info[index]
#         img = Image.open(path_img).convert('RGB')     # 0~255
#
#         if self.transform is not None:
#             img = self.transform(img)   # 在这里做transform，转为tensor等等
#
#         return img, label
#
#     def __len__(self):
#         return len(self.data_info)
#
#     @staticmethod
#     def get_img_info(data_dir):
#         data_info = list()
#         for root, dirs, _ in os.walk(data_dir):
#             # 遍历类别
#             for sub_dir in dirs:
#                 img_names = os.listdir(os.path.join(root, sub_dir))
#                 img_names = list(filter(lambda x: x.endswith('.jpg'), img_names))
#
#                 # 遍历图片
#                 for i in range(len(img_names)):
#                     img_name = img_names[i]
#                     path_img = os.path.join(root, sub_dir, img_name)
#                     label = rmb_label[sub_dir]
#                     data_info.append((path_img, int(label)))
#
#         return data_info
