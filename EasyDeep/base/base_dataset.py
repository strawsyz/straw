from torch.utils.data import Dataset

from base.base_logger import BaseLogger
from utils.config_utils import ConfigChecker
from utils.common_utils import copy_attr


class BaseDataSet(Dataset, BaseLogger, ConfigChecker):
    def __init__(self, config_instance=None):
        super(BaseDataSet, self).__init__()
        self.config_instance = config_instance
        self.load_config()

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def load_config(self):
        if self.config_instance is not None:
            copy_attr(self.config_instance, self)
        else:
            self.logger.error("not set a config file for dataset")
            raise NotImplementedError

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

    def get_samle_dataloader(self):
        """
        获得样本数据
        直接将数据保存在目标对象对应的地方
        :return:
        """
        raise NotImplementedError