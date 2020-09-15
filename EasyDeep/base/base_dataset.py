from abc import ABC, abstractmethod

from torch.utils.data import Dataset

from utils.common_utils import copy_attr
from utils.config_utils import ConfigChecker


class BaseDataSet(ABC, Dataset, ConfigChecker):
    def __init__(self, config_instance=None):
        super(BaseDataSet, self).__init__()
        self.config_instance = config_instance
        self.test_model = False
        # self.load_config()

    @abstractmethod
    def __len__(self):
        pass

    def __getitem__(self, index):
        pass

    # def load_config(self):
    #     if self.config_instance is not None:
    #         copy_attr(self.config_instance, self)
    #     else:
    #         self.logger.warning("not set a config file for dataset")

    def test(self):
        self.test_model = True

    def train(self):
        self.test_model = False

    def set_seed(self):
        import torch
        import numpy as np
        import random
        if self.random_state is not None:
            torch.manual_seed(self.random_state)  # cpu
            torch.cuda.manual_seed(self.random_state)  # gpu
            torch.cuda.manual_seed_all(self.random_state)
            np.random.seed(self.random_state)  # numpy
            random.seed(self.random_state)  # random and transforms
            torch.backends.cudnn.deterministic = True  # cudnn

    def get_sample_dataloader(self):
        """
        get sample data
        :return:
        """
        raise NotImplementedError

    @staticmethod
    def get_dataloader(self):
        pass
