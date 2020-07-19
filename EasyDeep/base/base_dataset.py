from torch.utils.data import Dataset

from base.base_logger import BaseLogger
from configs.net_config import NetConfig
from utils.config_utils import ConfigChecker
from utils.utils_ import copy_attr


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
            self.config_instance = NetConfig()
            copy_attr(self.config_instance, self)

    def test(self):
        self.test_model = True

    def train(self):
        self.test_model = False
