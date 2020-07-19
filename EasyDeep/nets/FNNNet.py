from base.base_net import BaseNet
from net_structures.BaseFNN import MyFNN


class FNNNet(BaseNet):
    def __init__(self, config_CLS=None):
        super(FNNNet, self).__init__(config_CLS)
        # if don't want to load basic net config file
        # then set config file before super()
        self.net = MyFNN


if __name__ == '__main__':
    base_net = FNNNet()
    base_net.get_net("1", False)
