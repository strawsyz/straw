from torch import nn
from torch import optim

from base.base_logger import BaseLogger
from configs.net_config import NetConfig
from utils.utils_ import copy_attr


class BaseNet(BaseLogger):

    def __init__(self, config_instance=None):
        self.config_instance = config_instance
        super(BaseNet, self).__init__()
        self.net = None
        self.loss_func_name = None
        self.optimizer = None
        self.scheduler = None
        self.is_use_cuda = None
        self.n_out = None
        self.load_config()

    def load_config(self):
        if self.config_instance is not None:
            copy_attr(self.config_instance, self)
        else:
            self.config_instance = NetConfig()
            copy_attr(self.config_instance, self)

    def get_net(self, target, is_use_gpu: bool):
        if self.net is None:
            self.logger.error("select a net_structure to be used")
        if self.loss_func_name == "BCEWithLogitsLoss":
            self.loss_function = nn.BCEWithLogitsLoss()
        elif self.loss_func_name == "MSE":
            self.loss_function = nn.MSELoss()
        else:
            self.logger.error("please set a valid loss function's name")
            raise RuntimeError
        if is_use_gpu:
            self.net = self.net.cuda()
            self.loss_function = self.loss_function.cuda()
        if self.optim_name == "adam":
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optim_name == "sgd":
            self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            self.logger.error("please set a valid loss function's name")
            raise RuntimeError
        if self.is_scheduler:
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.scheduler_step_size,
                                                       gamma=self.scheduler_gamma)
        self.copy_attr(target)

    def copy_attr(self, target):
        copy_attr(self, target)

    def unit_test(self):
        self.get_net(self, is_use_gpu=False)



if __name__ == '__main__':
    base_net = BaseNet()
    base_net.unit_test()
    # base_net.get_net(base_net, False)
