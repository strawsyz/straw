from base.base_net import BaseNet
from net_structures.FCN import FCN


class FCNNet(BaseNet):
    def __init__(self):
        # if don't want to load basic net config file
        # then set config file before super()
        super(FCNNet, self).__init__()
        self.net = FCN

if __name__ == '__main__':
    base_net = FCNNet()
    base_net.get_net("1", False)
