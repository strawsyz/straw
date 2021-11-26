from abc import ABC

from torch.utils.data import Dataset

from configs.dataset_config import BaseDataSetConfig


class BaseDataSet(BaseDataSetConfig, ABC, Dataset):
    def __init__(self):
        super(BaseDataSet, self).__init__()
        self.set_seed()

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def test(self):
        self.test_model = True

    def train(self):
        self.test_model = False

    def set_seed(self):
        import torch
        import numpy as np
        import random
        import os
        if self.random_state is not None:
            torch.manual_seed(self.random_state)  # cpu
            torch.cuda.manual_seed(self.random_state)  # gpu
            torch.cuda.manual_seed_all(self.random_state)
            np.random.seed(self.random_state)  # numpy
            random.seed(self.random_state)  # random and transforms
            torch.backends.cudnn.deterministic = True  # cudnn
            torch.backends.cudnn.benchmark = True
            os.environ['PYTHONHASHSEED'] = str(self.random_state)

    def get_sample_dataloader(self):
        """
        get sample data
        :return:
        """
        raise NotImplementedError

    @staticmethod
    def get_dataloader(self):
        pass

    def __str__(self):
        return __class__.__name__
