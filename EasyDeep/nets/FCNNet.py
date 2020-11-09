from base.base_net import BaseNet
from configs.net_config import FCNBaseNetConfig, FCNBaseNet4EdgeConfig, FCNResBaseConfig
from net_structures.FCN import FCN, FCN4Edge, FCNRes, FCNRes50

'''FCN networks used to image segmentation
type: 2D to 2D'''


class FCNNet(BaseNet, FCNBaseNetConfig):
    def __init__(self):
        super(FCNNet, self).__init__()
        self.net_structure = FCN(n_out=self.n_out, is_init=True)


class FCNResNet(BaseNet, FCNResBaseConfig):
    def __init__(self):
        super(FCNResNet, self).__init__()
        self.net_structure = FCNRes(is_init=self.is_init)


class FCN4EdgeNet(BaseNet, FCNBaseNet4EdgeConfig):
    def __init__(self):
        super(FCN4EdgeNet, self).__init__()
        self.net_structure = FCN4Edge(n_out=self.n_out, is_init=self.is_init)

class FCNRes50Net(BaseNet, FCNResBaseConfig):
    def __init__(self):
        super(FCNRes50Net, self).__init__()
        self.net_structure = FCNRes50(is_init=self.is_init)

