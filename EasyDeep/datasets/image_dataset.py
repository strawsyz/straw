import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from base.base_dataset import BaseDataSet
from configs.dataset_config import ImageDataSetConfig
from utils.utils_ import copy_attr


class ImageDataSet(BaseDataSet):
    def __init__(self, test_model=False):
        super(ImageDataSet, self).__init__()
        self.test_model = test_model
        self.image_paths = []
        self.mask_paths = []
        self.IMAGE_PATHS = []
        self.MASK_PATHS = []
        if self.random_state is not None:
            random.seed(self.random_state)
        file_names = sorted(os.listdir(self.image_path))
        if self.shuffle:
            random.shuffle(file_names)
        for file_name in file_names:
            self.IMAGE_PATHS.append(os.path.join(self.image_path, file_name))
            self.MASK_PATHS.append(os.path.join(self.mask_path, file_name))
            self.image_paths.append(os.path.join(self.image_path, file_name))
            self.mask_paths.append(os.path.join(self.mask_path, file_name))

    def get_dataloader(self, target):
        if self.random_state is not None:
            torch.manual_seed(self.random_state)  # cpu
            torch.cuda.manual_seed(self.random_state)  # gpu
            torch.cuda.manual_seed_all(self.random_state)
            np.random.seed(self.random_state)  # numpy
            random.seed(self.random_state)  # random and transforms
            torch.backends.cudnn.deterministic = True  # cudnn

        if not self.test_model:
            # self.train_data = self()
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
                self.val_loader = DataLoader(self.val_data, batch_size=self.batch_size, shuffle=True)
        else:
            # self.test_data = dataset(test_model=True)
            if self.num_test is not None:
                self.set_data_num(self.num_test)
            else:
                self.num_test = len(self)
            self.test_loader = DataLoader(self, batch_size=self.batch_size4test, shuffle=True)

        from utils.utils_ import copy_attr
        copy_attr(self, target)

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
        if self.test_model:
            return image, mask, os.path.basename(self.image_paths[index])
        else:
            return image, mask

    # todo 之后可能删除，如果randomstate不设置的话，就这样就无法区分测试集和训练集的位置了
    def set_data_num(self, num):
        if self.test_model:
            self.image_paths, self.mask_paths = self.IMAGE_PATHS[-num:], self.MASK_PATHS[-num:]
        else:
            self.image_paths, self.mask_paths = self.IMAGE_PATHS[:num], self.MASK_PATHS[:num]
