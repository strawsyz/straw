from base.base_logger import BaseLogger
from utils.config_utils import ConfigChecker
from utils.utils_ import copy_attr


class BaseExperiment(BaseLogger, ConfigChecker):

    def __init__(self, config_cls=None):
        super(BaseExperiment, self).__init__()
        # self.needed_config = None
        self.load_config(config_cls)

    def load_config(self, config_cls):
        if config_cls is not None:
            copy_attr(config_cls(), self)
        else:
            self.logger.error("need a eperiment config file!")

    def prepare_net(self):
        pass

    def prepare_data(self):
        pass

    def train(self):
        pass

    def train_one_epoch(self, epoch):
        pass

    def test(self):
        pass

    def estimate(self):
        pass

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
