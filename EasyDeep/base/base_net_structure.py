from torch import nn
from torch import optim

from utils.common_utils import copy_attr


class BaseNetStructure():

    def __init__(self):
        self.net = None
        self.loss_function = None
        self.optimizer = None
        self.scheduler = None
        self.is_use_cuda = None
        self.config = None
        self.load_config()
        pass

    def load_config(self):
        if hasattr(self, "config") and self.config is not None:
            copy_attr(self.config(), self)
        else:
            self.logger.error("please set config file for net")

    def get_net(self, is_use_gpu: bool):
        self.net = self.net(n_out=self.n_out)
        self.loss_function = nn.BCEWithLogitsLoss()
        if is_use_gpu:
            self.net = self.net.cuda()
            self.loss_function = self.loss_function.cuda()

        if self.optim_name == "adam":
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        else:
            self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.scheduler_step_size,
                                                   gamma=self.scheduler_gamma)

    def view_net_structure(self, x, filename=None):
        from torchviz import make_dot
        y = self(x)
        # g = make_dot(y)
        g = make_dot(y, params=dict(self.named_parameters()))
        # g = make_dot(y, params=dict(list(model.named_parameters()) + [('x', x)]))

        # 如果filename是None，就会生成一个 Digraph.gv.pdf 的PDF文件,用默认软件打开pdf文件
        # 否则按照设定的文件名进行保存
        g.view(filename=filename)

    def get_parameters_amount(net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}
