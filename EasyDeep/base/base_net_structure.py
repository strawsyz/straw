from utils.common_utils import copy_attr


class BaseNetStructure:

    def __init__(self):
        self.net = None
        self.loss_function = None
        self.optimizer = None
        self.scheduler = None
        self.is_use_cuda = None
        self.config = None
        self.load_config()

    def load_config(self):
        if hasattr(self, "config") and self.config is not None:
            copy_attr(self.config(), self)
        else:
            self.logger.error("please set config file for net")

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
        for name, parameters in net.named_parameters():
            print(name, ':', parameters.size())
        return {'Total': total_num, 'Trainable': trainable_num}

    def print_layer_size(self):
        """print every layer's parameter_size"""
        for name, parameters in self.named_parameters():
            print(name, ':', parameters.size())
            # parm[name] = parameters.detach().numpy()

    def describe(self):
        self.print_layer_size()
        self.print_parameters_amount()
