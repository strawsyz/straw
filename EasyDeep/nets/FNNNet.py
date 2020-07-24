from base.base_net import BaseNet
from net_structures.BaseFNN import MyFNN


class FNNNet(BaseNet):
    def __init__(self, config_instance=None):
        super(FNNNet, self).__init__(config_instance)
        self.net = MyFNN(n_in=self.n_in, n_out=self.n_out)


if __name__ == '__main__':
    net = FNNNet()
    net.unit_test()
