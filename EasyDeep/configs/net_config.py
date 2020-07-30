class NetConfig:
    def __init__(self):
        # net structure config
        self.n_out = 1
        # other net config
        self.loss_func_name = "BCEWithLogitsLoss"
        self.optim_name = "adam"
        self.is_scheduler = True
        self.scheduler_step_size = 15
        self.scheduler_gamma = 0.8
        self.lr = 0.002
        self.weight_decay = 0.001


class FCNNetConfig(NetConfig):
    def __init__(self):
        super(FCNNetConfig, self).__init__()


class FNNNetConfig(NetConfig):
    def __init__(self):
        super(FNNNetConfig, self).__init__()
        self.loss_func_name = "MSE"
        self.n_in = 887
        self.n_out = 1


class CNN1DNetConfig(NetConfig):
    def __init__(self):
        super(CNN1DNetConfig, self).__init__()
        self.loss_func_name = "MSE"