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
