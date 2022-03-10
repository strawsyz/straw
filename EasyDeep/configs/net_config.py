# need to rename db_config_example.py to db_config.py
from Args.video_args import get_args
from base.base_config import BaseNetConfig


class FCNVgg16NetConfig(BaseNetConfig):
    def __init__(self):
        super(FCNVgg16NetConfig, self).__init__()
        self.n_out = 1
        # if init parameters in the network
        self.is_init = True
        self.lr = 0.0003
        self.pretrained = True
        # self.loss_func_name = "BCEWithLogitsLoss"
        self.loss_func_name = "DiceLoss"
        self.optim_name = "adam"
        self.is_scheduler = True
        self.scheduler_step_size = 15
        self.scheduler_gamma = 0.8
        self.weight_decay = None
        self.init_attr()




class FCNResBaseConfig(BaseNetConfig):
    def __init__(self):
        super(FCNResBaseConfig, self).__init__()
        self.is_init = True
        self.lr = 0.0003
        self.loss_func_name = "BCEWithLogitsLoss"
        # self.loss_func_name = "DiceLoss"
        self.optim_name = "adam"
        self.is_scheduler = True
        self.scheduler_step_size = 15
        self.scheduler_gamma = 0.8
        self.weight_decay = None
        self.init_attr()


class FCNBaseNet4EdgeConfig(BaseNetConfig):
    def __init__(self):
        super(FCNBaseNet4EdgeConfig, self).__init__()
        self.n_in = 4
        self.n_out = 1
        # if init parameters in the network
        self.is_init = True
        self.lr = 0.0003
        self.pretrained = True
        self.loss_func_name = "DiceLoss"
        # self.loss_func_name = "DiceBCELoss"
        self.weight_decay = 0.01
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

class MultiLabelNetConfig(BaseNetConfig):
    def __init__(self):
        super(MultiLabelNetConfig, self).__init__()
        self.n_in = 46
        self.n_out = 11
        self.loss_func_name = "BCEWithLogitsLoss"
        self.scheduler_step_size = 20
        self.optim_name = "Adam"
        self.lr = 0.001


class VideoFeatureNetConfig(BaseNetConfig):
    def __init__(self):
        super(VideoFeatureNetConfig, self).__init__()
        self.n_in = 2048
        self.embeddings_dim = 512
        self.n_layer = 6
        self.heads = 1
        self.n_out = 101
        self.loss_func_name = "BCEWithLogitsLoss"
        self.scheduler_step_size = 20
        self.optim_name = "Adam"
        self.lr = 0.001

        # set According to the Args
        args = get_args()
        self.lr = args.LR
        # self.n_layer = args.n_layers
        self.heads = args.heads
        self.embedding_dim = args.embeddings_dim
        self.N = args.n_layers
        self.vocab_size = args.vocab_size
        self.model_name = args.model_name
