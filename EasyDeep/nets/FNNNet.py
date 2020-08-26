from base.base_net import BaseNet
from net_structures.FNN import MyFNN
from configs.net_config import FNNNetConfig


class FNNNet(FNNNetConfig, BaseNet):
    def __init__(self):
        super(FNNNet, self).__init__()
        self.net = MyFNN(n_in=self.n_in, n_out=self.n_out)


