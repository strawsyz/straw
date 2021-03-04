# need to rename db_config_example.py to db_config.py
from base.base_config import BaseNetConfig


class FCNVgg16NetConfig(BaseNetConfig):
    def __init__(self):
        super(FCNVgg16NetConfig, self).__init__()
        self.n_out = 1
        # if init parameters in the network
        self.is_init = True
        self.lr = 0.0001
        self.pretrained = True
        self.loss_func_name = "BCEWithLogitsLoss"


class FCNBaseNet4EdgeConfig(BaseNetConfig):
    def __init__(self):
        super(FCNBaseNet4EdgeConfig, self).__init__()
        self.n_in = 4
        self.n_out = 1
        # if init parameters in the network
        self.is_init = True
        self.lr = 0.001
        self.pretrained = True
        self.loss_func_name = "BCEWithLogitsLoss"


class FCNResBaseConfig(BaseNetConfig):
    def __init__(self):
        super(FCNResBaseConfig, self).__init__()
        self.is_init = True
        self.lr = 0.001


class FNNBaseNetConfig(BaseNetConfig):
    def __init__(self):
        super(FNNBaseNetConfig, self).__init__()
        self.loss_func_name = "MSE"
        self.n_in = 5000
        self.n_out = 1
        self.weight_decay = 0.005


class CNN1DBaseNetConfig(BaseNetConfig):
    def __init__(self):
        super(CNN1DBaseNetConfig, self).__init__()
        self.loss_func_name = "MSE"


class MnistConfigBase(BaseNetConfig):
    def __init__(self):
        super(MnistConfigBase, self).__init__()
        self.n_in = 28 * 28
        self.n_out = 10
        self.loss_func_name = "MSE"


class LoanNetConfig(BaseNetConfig):
    def __init__(self):
        super(LoanNetConfig, self).__init__()
        self.n_in = 4
        self.n_out = 1
        self.loss_func_name = "MSE"
        self.scheduler_step_size = 20
        self.optim_name = "sgd"
        self.lr = 0.001
