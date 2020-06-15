# -*- coding: utf-8 -*-

import os
import random

import cv2
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset


class PolypDataset(Dataset):
    def __init__(self, image_path, mask_path, image_transforms=None, mask_transforms=None, test=False):
        super(PolypDataset, self).__init__()
        self.image_paths = []
        self.mask_paths = []
        self.IMAGE_PATHS = []
        self.MASK_PATHS = []
        self.transforms = image_transforms
        self.mask_transforms = mask_transforms
        # 随机排序
        random.seed(7)
        image_paths = sorted(os.listdir(image_path))
        random.shuffle(image_paths)
        for file_name in image_paths:
            self.IMAGE_PATHS.append(os.path.join(image_path, file_name))
            self.MASK_PATHS.append(os.path.join(mask_path, file_name))
            self.image_paths.append(os.path.join(image_path, file_name))
            self.mask_paths.append(os.path.join(mask_path, file_name))
        self.test = test

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        mask = Image.open(self.mask_paths[index])
        # 对图像进行处理
        if self.transforms is not None:
            image = self.transforms(image)
        if self.mask_transforms is not None:
            # todo 存在一些不是0也不是1的数字，非常小，对结果应该没有太大影响，但姑且记录一下
            mask = self.mask_transforms(mask)
            # 要对数据二值化
            # 应该不需要对mask图像做处理
            # label = self.transforms(label)
        # sample = self.transform(image, label)
        if self.test:
            # 如果是测试模型，就会返回图像的名字
            return image, mask, os.path.basename(self.image_paths[index])
        else:
            return image, mask

    def set_data_num(self, num):
        # self.image_paths, self.mask_paths = self.IMAGE_PATHS[:num], self.MASK_PATHS[:num]
        if self.test:
            # 如果是训练模式，就从图像的后面往前选取num个图像
            self.image_paths, self.mask_paths = self.IMAGE_PATHS[-num:], self.MASK_PATHS[-num:]
            pass
        else:
            # 设置数据集的大小
            self.image_paths, self.mask_paths = self.IMAGE_PATHS[:num], self.MASK_PATHS[:num]


class NYU(Dataset):
    """
    NYU数据集的处理
    https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html
    """

    def __init__(self, image_path, label_path, transform=None, width=213, height=213):
        if transform:
            # 如果有其他指定数据处理函数就使用其他的函数
            self.transform = transform

        self.IMAGE_PATH = image_path
        self.LABEL_PATH = label_path
        self.set_dataset()

        self.width = width
        self.height = height

    def __len__(self):
        return len(os.listdir(self.IMAGE_PATH))

    def set_dataset(self):
        """设置图像的路径列表"""
        image_paths = []
        for file_name in os.listdir(self.IMAGE_PATH):
            image_paths.append(os.path.join(self.IMAGE_PATH, file_name))
        image_paths.sort()
        self.image_paths = image_paths

        label_paths = []
        for file_name in os.listdir(self.LABEL_PATH):
            label_paths.append(os.path.join(self.LABEL_PATH, file_name))
        label_paths.sort()
        self.label_paths = image_paths

    def __getitem__(self, index):

        image = Image.open(self.image_paths[index])
        label = Image.open(self.label_paths[index])
        # 对图像进行处理
        sample = self.img_transform(image, label)
        return sample

    def img_transform(self, image, label):  # 将原图像和标签图像都进行处理
        image = cv2.resize(image, (self.height, self.width), interpolation=cv2.INTER_LINEAR)
        # todo 还有很多缩放方法可以选择，有空再仔细研究一次
        label = cv2.resize(label, (self.height, self.width), interpolation=cv2.INTER_LINEAR)
        # numpy ：HxWxC
        # torch ：CxHxW
        # 转换通道的位置
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image).float()
        image = image / 255
        # 标准化
        image = torchvision.transforms.Normalize(mean=[0.482, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])(image)
        sample = {"image": image,
                  "label": torch.from_numpy(label).float()}

        # 输出样本数据
        return sample
