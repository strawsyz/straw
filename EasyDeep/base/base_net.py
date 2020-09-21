from torch import nn
from torch import optim

from base.base_logger import BaseLogger
from utils.common_utils import copy_attr


class BaseNet(BaseLogger):

    def __init__(self):
        self.loss_func_name = None
        self.optim_name = None
        self.weight_decay = None
        self.lr = None
        self.is_scheduler = False
        self.scheduler_step_size = None
        self.scheduler_gamma = 0.1
        self.net_structure = None
        super(BaseNet, self).__init__()

    def get_net(self, target, is_use_gpu: bool):
        if self.net_structure is None:
            self.logger.error("please select a net_structure to use")
        self.set_loss_function()
        if is_use_gpu:
            self.net_structure = self.net_structure.cuda()
            self.loss_function = self.loss_function.cuda()
        self.set_optimizer()
        self.set_scheduler()
        self.copy_attr(target)

    def unit_test(self, data):
        self.get_net(self, is_use_gpu=False)
        return self.net_structure(data)

    def set_loss_function(self):
        if self.loss_func_name == "BCEWithLogitsLoss":
            self.loss_function = nn.BCEWithLogitsLoss()
        elif self.loss_func_name == "MSE":
            self.loss_function = nn.MSELoss()
        elif self.loss_func_name == "L1Loss":
            self.loss_function = nn.L1Loss()
        elif self.loss_func_name == "CrossEntropyLoss":
            self.loss_function = nn.CrossEntropyLoss()
        elif self.loss_func_name == "NLLLoss":
            self.loss_function = nn.NLLLoss()
        elif self.loss_func_name == "PoissonNLLLoss":
            self.loss_function = nn.PoissonNLLLoss()
        elif self.loss_func_name == "KLDivLoss":
            self.loss_function = nn.KLDivLoss()
        elif self.loss_func_name == "BCELoss":
            self.loss_function = nn.BCELoss()
        elif self.loss_func_name == "MarginRankingLoss":
            self.loss_function = nn.MarginRankingLoss()
        elif self.loss_func_name == "HingeEmbeddingLoss":
            self.loss_function = nn.HingeEmbeddingLoss()
        elif self.loss_func_name == "MultiLabelMarginLoss":
            self.loss_function = nn.MultiLabelMarginLoss()
        elif self.loss_func_name == "SmoothL1loss":
            self.loss_function = nn.SmoothL1Loss()
        elif self.loss_func_name == "SoftMarginLoss":
            self.loss_function = nn.SoftMarginLoss()
        elif self.loss_func_name == "MultiLabelSoftMarginLoss":
            self.loss_function = nn.MultiLabelSoftMarginLoss()
        elif self.loss_func_name == "CosineEmbeddingLoss":
            self.loss_function = nn.CosineEmbeddingLoss()
        elif self.loss_func_name == "MultiMarginLoss":
            self.loss_function = nn.MultiMarginLoss()
        elif self.loss_func_name == "TripletMarginLoss":
            self.loss_function = nn.TripletMarginLoss()
        elif self.loss_func_name == "CTCLoss":
            self.loss_function = nn.CTCLoss()
        else:
            self.logger.error("please set a valid loss function's name")
            raise RuntimeError("please set a valid loss function's name")

    def set_optimizer(self):
        if self.optim_name == "adam":
            if self.weight_decay is None:
                self.optimizer = optim.Adam(self.net_structure.parameters(), lr=self.lr)
            else:
                self.optimizer = optim.Adam(self.net_structure.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optim_name == "sgd":
            if self.weight_decay is None:
                self.optimizer = optim.Adam(self.net_structure.parameters(), lr=self.lr)
            else:
                self.optimizer = optim.SGD(self.net_structure.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            self.logger.error("please set a optimizer's name")
            raise RuntimeError

    def set_scheduler(self):
        if self.is_scheduler:
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.scheduler_step_size,
                                                       gamma=self.scheduler_gamma)

    def view_net_structure(self, input_data, filename=None):
        """output the structure of net as pdf file"""
        return self.net_structure.view_net_structure(input_data, filename)

    def get_parameters_amount(self):
        return self.net_structure.getget_parameters_amount()

    def copy_attr(self, target):
        # todo decrease amount of parameters to copy
        copy_attr(self, target)

    def __str__(self):
        return __class__.__name__
