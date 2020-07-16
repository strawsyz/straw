from torch.utils.data import Dataset
from base.base_logger import BaseLogger
from utils.config_utils import ConfigChecker
from utils.utils_ import copy_attr


class BaseDataSet(Dataset, BaseLogger, ConfigChecker):
    def __init__(self, config_cls=None):
        super(BaseDataSet, self).__init__()
        self.config_cls = config_cls
        self.load_config()

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def load_config(self):
        if self.config_cls is not None:
            copy_attr(self.config_cls(), self)
        else:
            raise NotImplementedError
