from torch import nn
from torch import optim

from base.base_logger import BaseLogger
from configs.net_config import NetConfig
from utils.utils_ import copy_attr


class BaseNet(BaseLogger):
    __slots__ = "net"
    def __init__(self, config=None):
        self.config = config
        super(BaseNet, self).__init__()
        self.net = None
        self.loss_func_name = None
        self.optimizer = None
        self.scheduler = None
        self.is_use_cuda = None
        self.n_out = None
        if self.config is None:
            self.config = NetConfig
        self.load_config()

    def load_config(self):
        if hasattr(self, "config") and self.config is not None:
            copy_attr(self.config(), self)
        else:
            self.logger.error("please set config file for net")

    def get_net(self, target, is_use_gpu: bool):
        if self.net is None:
            self.logger.error("select a net_structure to be used")
        self.net = self.net(n_out=self.n_out)
        if self.loss_func_name == "BCEWithLogitsLoss":
            self.loss_function = nn.BCEWithLogitsLoss()
        else:
            self.logger.error("please set a valid loss function's name")
        if is_use_gpu:
            self.net = self.net.cuda()
            self.loss_function = self.loss_function.cuda()

        if self.optim_name == "adam":
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr)
        if self.is_scheduler:
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.scheduler_step_size,
                                                       gamma=self.scheduler_gamma)
        # todo 不知道为什么，使用__dict__不会显示net属性
        target.net = self.net
        copy_attr(self, target)


if __name__ == '__main__':
    base_net = BaseNet()
    base_net.get_net("1", False)
