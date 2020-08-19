from base.base_logger import BaseLogger
from utils.config_utils import ConfigChecker
from utils.common_utils import copy_attr


class BaseExperiment(BaseLogger, ConfigChecker):

    def __init__(self, config_instance=None):
        super(BaseExperiment, self).__init__()
        self.config_instance = config_instance
        self.load_config()

    def load_config(self):
        if self.config_instance is not None:
            copy_attr(self.config_instance, self)
        else:
            self.logger.error("need a eperiment config file!")
            raise NotImplementedError

    def list_config(self):
        """list all config on this experiment"""
        if not hasattr(self, "config_instance") or self.config_instance is None:
            self.logger.warning("not set a configure file for the experiment")
            raise RuntimeError
        else:
            self.logger.info(str(self.config_instance))

    def prepare_net(self):
        pass

    def prepare_dataset(self):
        pass

    def train(self):
        raise NotImplementedError

    def train_one_epoch(self, epoch):
        pass

    def test(self):
        raise NotImplementedError

    def estimate(self):
        raise NotImplementedError

    def save(self):
        # 保存模型
        # 保存其他参数
        pass

    def load(self):
        pass

    def show_history(self):
        raise NotImplementedError

    def check(self):
        # 检查输入图像的形状
        # 配置文件是否正确
        raise NotImplementedError


if __name__ == '__main__':
    expriment = BaseExperiment()
    expriment.train()
    expriment.show_history()
