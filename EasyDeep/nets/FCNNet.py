from base.base_net import BaseNet
from net_structures.FCN import FCN, FCN4Edge
from configs.net_config import FCNNetConfig, FCNNet4EdgeConfig


class FCNNet(BaseNet, FCNNetConfig):
    def __init__(self, config_instance=None):
        super(FCNNet, self).__init__(config_instance)
        self.net = FCN(n_out=self.n_out, is_init=True)


class FCN4EdgeNet(BaseNet, FCNNet4EdgeConfig):
    def __init__(self, config_instance=None):
        super(FCN4EdgeNet, self).__init__(config_instance)
        self.net = FCN4Edge(n_out=self.n_out, is_init=self.is_init)


if __name__ == '__main__':
    net = FCNNet()
    net.unit_test()
