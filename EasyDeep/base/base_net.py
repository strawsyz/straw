class BaseNet:

    def __init__(self):
        self.net = None
        self.loss_function = None
        self.optimizer = None
        self.scheduler = None
        self.is_use_cuda = None
        self.load_config()
        pass

    def load_config(self):
        pass
