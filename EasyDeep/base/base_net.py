from torch import nn
from torch import optim
import torch
from base.base_logger import BaseLogger
from utils.common_utils import copy_attr


class BaseNet(BaseLogger):

    def __init__(self):
        super(BaseNet, self).__init__()

    def get_net(self, target, is_use_gpu: bool):
        if self.net is None:
            self.logger.error("select a net_structure to be used")
        self.set_loss_function()
        if is_use_gpu:
            self.net = self.net.cuda()
            self.loss_function = self.loss_function.cuda()
        self.set_optimizer()
        self.set_scheduler()
        self.copy_attr(target)

    def unit_test(self, data=torch.randn(40000)):
        self.get_net(self, is_use_gpu=False)
        return self.net(data)

    def set_loss_function(self):
        if self.loss_func_name == "BCEWithLogitsLoss":
            self.loss_function = nn.BCEWithLogitsLoss()
        elif self.loss_func_name == "MSE":
            self.loss_function = nn.MSELoss()
        else:
            self.logger.error("please set a valid loss function's name")
            raise RuntimeError("please set a valid loss function's name")

    def set_optimizer(self):
        if self.optim_name == "adam":
            if self.weight_decay is None:
                self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
            else:
                self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optim_name == "sgd":
            if self.weight_decay is None:
                self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
            else:
                self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            self.logger.error("please set a optimizer's name")
            raise RuntimeError

    def set_scheduler(self):
        if self.is_scheduler:
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.scheduler_step_size,
                                                       # todo self.scheduler_gamma默认是0.1
                                                       gamma=self.scheduler_gamma)

    def view_net_structure(self, input_data, filename=None):
        """output the structure of net"""
        return self.net.view_net_structure(input_data, filename)

    def get_parameters_amount(self):
        return self.net.getget_parameters_amount()

    def copy_attr(self, target):
        # todo 整理一下，减少复制的参数
        copy_attr(self, target)


if __name__ == '__main__':
    base_net = BaseNet()
    base_net.unit_test()
    # base_net.get_net(base_net, False)
