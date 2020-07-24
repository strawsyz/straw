from base.base_net import BaseNet
from net_structures.FCN import FCN


class FCNNet(BaseNet):
    def __init__(self, config_instance=None):
        super(FCNNet, self).__init__(config_instance)
        self.net = FCN(n_out=self.n_out, is_init=True)


if __name__ == '__main__':
    net = FCNNet()
    net.unit_test()
