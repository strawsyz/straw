# -*- coding: utf-8 -*-

import os
import torch
from torch.utils.data import Dataset
import cv2
import torchvision
from PIL import Image


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
        sample = self.transform(image, label)
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
