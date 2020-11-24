import os
import random

import torch
from PIL import Image
from torch.utils.data import DataLoader

from base.base_dataset import BaseDataSet
from configs.dataset_config import ImageDataSetConfig
from utils.common_utils import copy_attr
from utils.common_utils import copy_need_attr

"""
mask图像和原图像两个文件夹
根据文件名判断是否是同一个文件夹
"""


class ImageDataSet(BaseDataSet, ImageDataSetConfig):

    def __init__(self):
        super(ImageDataSet, self).__init__()
        self.image_paths = []
        self.mask_paths = []
        self.IMAGE_PATHS = []
        self.MASK_PATHS = []
        self.set_seed()
        file_names = sorted(os.listdir(self.image_path))
        if self.shuffle:
            random.shuffle(file_names)
        for file_name in file_names:
            self.IMAGE_PATHS.append(os.path.join(self.image_path, file_name))
            self.MASK_PATHS.append(os.path.join(self.mask_path, file_name))
            self.image_paths.append(os.path.join(self.image_path, file_name))
            self.mask_paths.append(os.path.join(self.mask_path, file_name))
        if self.num_train is None:
            self.num_train = len(self.mask_paths)
        if self.num_test is None:
            self.num_test = len(self.mask_paths)

    def get_sample_dataloader(self, num_samples, target):
        """get sample dataloader to test"""
        self.image_paths, self.mask_paths = self.IMAGE_PATHS[:num_samples * 3], self.MASK_PATHS[:num_samples * 3]
        self.train_data, self.valid_data, self.test_data = torch.utils.data.random_split(self,
                                                                                         [num_samples,
                                                                                          num_samples,
                                                                                          num_samples])
        self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        self.valid_loader = DataLoader(self.valid_data, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_data, batch_size=self.batch_size4test, shuffle=True)
        copy_need_attr(target, ["valid_loader", "train_loader", "test_loader"])

    def train(self):
        super().train()

    def get_dataloader(self, target):
        if not self.test_model:
            if self.num_train is not None:
                if len(self) > self.num_train:
                    self.set_data_num(self.num_train)
                else:
                    self.num_train = len(self)
            else:
                self.num_train = len(self.train_data)
            if self.valid_rate is None:
                self.train_loader = DataLoader(self, batch_size=self.batch_size, shuffle=True)
                return self.train_loader
            else:
                num_valid_data = int(self.num_train * self.valid_rate)
                if num_valid_data == 0 or num_valid_data == self.num_train:
                    self.logger.error("valid datateset is None or train dataset is None")
                self.train_data, self.val_data = torch.utils.data.random_split(self,
                                                                               [self.num_train - num_valid_data,
                                                                                num_valid_data])
                self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
                self.valid_loader = DataLoader(self.val_data, batch_size=self.batch_size, shuffle=True)
        else:
            if self.num_test is not None:
                self.set_data_num(self.num_test)
            else:
                self.num_test = len(self)
            self.test_loader = DataLoader(self, batch_size=self.batch_size4test, shuffle=True)

        copy_attr(self, target)

    def sort_dataset(self):
        """calculate number of train_dataset, valid_dataset and test_dataset"""
        # calculate the number of samples to train
        if not self.test_model:
            if self.num_train is not None:
                if len(self) > self.num_train:
                    self.set_data_num(self.num_train)
                else:
                    self.num_train = len(self)
            else:
                self.num_train = len(self.train_data)
            if self.valid_rate is None:
                self.num_valid = 0
            else:
                num_valid_data = int(self.num_train * self.valid_rate)
                if num_valid_data == 0 or num_valid_data == self.num_train:
                    self.logger.error("valid datateset is None or train dataset is None")
                self.train_data, self.val_data = torch.utils.data.random_split(self,
                                                                               [self.num_train - num_valid_data,
                                                                                num_valid_data])
                self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
                self.valid_loader = DataLoader(self.val_data, batch_size=self.batch_size, shuffle=True)
        else:
            if self.num_test is not None:
                self.set_data_num(self.num_test)
            else:
                self.num_test = len(self)
            self.test_loader = DataLoader(self, batch_size=self.batch_size4test, shuffle=True)


        if self.test_model:
            if self.num_test is not None:
                self.set_data_num(self.num_test)
            else:
                self.num_test = len(self)
        else:
            if self.num_train is None:
                self.num_train = len(self.train_data)
            else:
                if len(self) > self.num_train:
                    self.set_data_num(self.num_train)
                else:
                    self.num_train = len(self)
        if self.num_train is not None:
            # if number of train which been set is bigger than amount of train_dataset
            if len(self.Y) > self.num_train:
                self.set_data_num(self.num_train)
            else:
                self.num_train = len(self)
        else:
            self.num_train = len(self.Y)

        # calculate the number of samples to valid
        if self.valid_rate is None:
            self.num_valid = 0
        else:
            self.num_valid = int(self.num_train * self.valid_rate)
            self.num_train = self.num_train - self.num_valid
            if self.num_valid == 0 or self.num_valid == self.num_train:
                self.logger.error("valid dataset is None or train dataset is None")


        self.logger.info(
            "num_train:{} \t num_valid:{} \t num_test:{} ".format(self.num_train, self.num_valid, self.num_test))
        return self.num_train, self.num_test, self.num_valid

    def create_dataset(self):
        pass

    def create_dataloader(self):
        pass

    def load_config(self):
        copy_attr(ImageDataSetConfig(), self)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        mask = Image.open(self.mask_paths[index])
        if self.image_transforms is not None:
            image = self.image_transforms(image)
        if self.mask_transforms is not None:
            mask = self.mask_transforms(mask)
        # if use few sample for test ,will not have test_model
        if getattr(self, "test_model", False):
            return image, mask, os.path.basename(self.image_paths[index])
        else:
            return image, mask

    def set_data_num(self, num):
        if self.test_model:
            self.image_paths, self.mask_paths = self.IMAGE_PATHS[-num:], self.MASK_PATHS[-num:]
        else:
            self.image_paths, self.mask_paths = self.IMAGE_PATHS[:num], self.MASK_PATHS[:num]


from configs.dataset_config import ImageDataSet4EdgeConfig


class ImageDataSet4Edge(BaseDataSet, ImageDataSet4EdgeConfig):

    def __init__(self, ):
        super(ImageDataSet4Edge, self).__init__()
        self.image_paths = []
        self.mask_paths = []
        self.IMAGE_PATHS = []
        self.MASK_PATHS = []
        self.PREDICT_PATHS = []
        self.EDGE_PATHS = []
        self.edge_paths = []
        self.predict_paths = []
        self.set_seed()
        file_names = sorted(os.listdir(self.image_path))
        if self.shuffle:
            random.shuffle(file_names)
        for file_name in file_names:
            self.IMAGE_PATHS.append(os.path.join(self.image_path, file_name))
            self.MASK_PATHS.append(os.path.join(self.mask_path, file_name))
            self.EDGE_PATHS.append(os.path.join(self.edge_path, file_name))
            self.PREDICT_PATHS.append(os.path.join(self.predict_path, file_name))
            self.image_paths.append(os.path.join(self.image_path, file_name))
            self.mask_paths.append(os.path.join(self.mask_path, file_name))
            self.edge_paths.append(os.path.join(self.edge_path, file_name))
            self.predict_paths.append(os.path.join(self.predict_path, file_name))

    def copy_attr(self, target, attr_names):
        for attr_name in attr_names:
            setattr(target, attr_name, getattr(self, attr_name))

    def get_dataloader(self, target):
        if not self.test_model:
            if self.num_train is not None:
                if len(self) > self.num_train:
                    self.set_data_num(self.num_train)
                else:
                    self.num_train = len(self)
            else:
                self.num_train = len(self.train_data)
            if self.valid_rate is None:
                self.train_loader = DataLoader(self, batch_size=self.batch_size, shuffle=True)
                return self.train_loader
            else:
                num_valid_data = int(self.num_train * self.valid_rate)
                if num_valid_data == 0 or num_valid_data == self.num_train:
                    self.logger.error("valid datateset is None or train dataset is None")
                self.train_data, self.val_data = torch.utils.data.random_split(self,
                                                                               [self.num_train - num_valid_data,
                                                                                num_valid_data])
                self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
                self.valid_loader = DataLoader(self.val_data, batch_size=self.batch_size, shuffle=True)
        else:
            if self.num_test is not None:
                self.set_data_num(self.num_test)
            else:
                self.num_test = len(self)
            self.test_loader = DataLoader(self, batch_size=self.batch_size4test, shuffle=True)

        from utils.common_utils import copy_attr
        copy_attr(self, target)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        edge_path = self.edge_paths[index]
        mask_path = self.mask_paths[index]
        predict_path = self.predict_paths[index]
        image = Image.open(image_path)
        edge = Image.open(edge_path)
        mask = Image.open(mask_path)
        predict = Image.open(predict_path)
        if self.image_transforms is not None:
            image = self.image_transforms(image)
        if self.mask_transforms is not None:
            mask = self.mask_transforms(mask)
        if self.edge_transforms is not None:
            edge = self.edge_transforms(edge)
        if self.predict_transforms is not None:
            predict = self.predict_transforms(predict)

        # concate source image and edge image to create X data
        image = torch.cat([image, edge, predict], dim=0)
        # image = torch.cat([image, predict], dim=0)
        # if use few sample for test ,will not have test_model
        if getattr(self, "test_model", False):
            return image, mask, os.path.basename(self.image_paths[index])
        else:
            return image, mask

    def set_data_num(self, num):
        if self.test_model:
            self.image_paths, self.mask_paths = self.IMAGE_PATHS[-num:], self.MASK_PATHS[-num:]
        else:
            self.image_paths, self.mask_paths = self.IMAGE_PATHS[:num], self.MASK_PATHS[:num]
