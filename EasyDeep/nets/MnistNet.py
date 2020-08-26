from base.base_net import BaseNet
from net_structures.FNN import AdaptiveFNN
from configs.net_config import MnistConfig


class MnistNet(MnistConfig, BaseNet):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.net = AdaptiveFNN(num_in=self.n_in, num_out=self.n_out)
        # print(self.n_in)
