from configs.experiment_config import BaseExperimentConfig


class BaseExperiment(BaseExperimentConfig):

    def __init__(self):
        super(BaseExperiment, self).__init__()
        # self.config_instance = config_instance
        # self.load_config()

    # def load_config(self):
    #     if self.config_instance is not None:
    #         copy_attr(self.config_instance, self)
    #     else:
    #         self.logger.error("need a eperiment config file!")
    #         raise NotImplementedError

    # def show_config(self):
    #     """list all configure in a experiment"""
    #     # if self.config_instance is None:
    #     #     self.logger.warning("please set a configure file for the experiment")
    #     #     raise RuntimeError("please set a configure file for the experiment")
    #     # else:
    #     config_instance_str = str(self.config_instance)
    #     self.logger.info(config_instance_str)
    #     return config_instance_str

    def prepare_net(self):
        raise NotImplementedError

    def prepare_dataset(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def before_test(self):
        raise NotImplementedError

    def estimate(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def estimate_history(self):
        raise NotImplementedError

    def check(self):
        # used to check config file and something else
        raise NotImplementedError


if __name__ == '__main__':
    expriment = BaseExperiment()
    expriment.train()
    expriment.estimate_history()
