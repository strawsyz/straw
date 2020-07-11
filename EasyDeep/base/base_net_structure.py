from base_logger import BaseLogger
from torch import nn
from torch import optim

from utils.utils_ import copy_attr


class BaseNetStructure(BaseLogger):

    def __init__(self):
        self.net = None
        self.loss_function = None
        self.optimizer = None
        self.scheduler = None
        self.is_use_cuda = None
        self.config = None
        self.load_config()
        pass

    def load_config(self):
        if hasattr(self, "config") and self.config is not None:
            copy_attr(self.config(), self)
        else:
            self.logger.error("please set config file for net")

    def get_net(self, is_use_gpu: bool):
        self.net = self.net(n_out=self.n_out)
        self.loss_function = nn.BCEWithLogitsLoss()
        if is_use_gpu:
            self.net = self.net.cuda()
            self.loss_function = self.loss_function.cuda()

        if self.optim_name == "adam":
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        else:
            self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.scheduler_step_size,
                                                   gamma=self.scheduler_gamma)
        return
