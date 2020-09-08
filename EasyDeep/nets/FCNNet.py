from base.base_net import BaseNet
from configs.net_config import FCNNetConfig, FCNNet4EdgeConfig, FCNResConfig
from net_structures.FCN import FCN, FCN4Edge, FCNRes


class FCNNet(BaseNet, FCNNetConfig):
    def __init__(self):
        super(FCNNet, self).__init__()
        self.net = FCN(n_out=self.n_out, is_init=True)


class FCNResNet(BaseNet, FCNResConfig):
    def __init__(self):
        super(FCNResNet, self).__init__()
        self.net = FCNRes(is_init=self.is_init)


class FCN4EdgeNet(BaseNet, FCNNet4EdgeConfig):
    def __init__(self):
        super(FCN4EdgeNet, self).__init__()
        self.net = FCN4Edge(n_out=self.n_out, is_init=self.is_init)
