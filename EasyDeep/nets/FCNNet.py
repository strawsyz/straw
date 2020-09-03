from base.base_net import BaseNet
from configs.net_config import FCNNetConfig, FCNNet4EdgeConfig
from net_structures.FCN import FCN, FCN4Edge


class FCNNet(BaseNet, FCNNetConfig):
    def __init__(self):
        super(FCNNet, self).__init__()
        self.net = FCN(n_out=self.n_out, is_init=True)


class FCN4EdgeNet(BaseNet, FCNNet4EdgeConfig):
    def __init__(self):
        super(FCN4EdgeNet, self).__init__()
        self.net = FCN4Edge(n_out=self.n_out, is_init=self.is_init)

