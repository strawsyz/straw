import os
import random

from PIL import Image

from base.base_dataset import BaseDataSet
from configs.dataset_config import ImageDataSetConfig
from utils.utils_ import copy_attr


class ImageDataSet(BaseDataSet):
    def __init__(self, test_model=False):
        super(ImageDataSet, self).__init__()
        self.load_config()
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
