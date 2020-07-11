class NetConfig:
    def __init__(self):
        self.net = None
        self.loss_func_name = "BCEWithLogitsLoss"
        self.optim_name = "adam"
        self.is_scheduler = True
        self.scheduler_step_size = 15
        self.scheduler_gamma = 0.8
        self.n_out = 1
        self.lr = 0.002
