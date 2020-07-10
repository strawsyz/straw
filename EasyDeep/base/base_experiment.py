from base.base_logger import BaseLogger


class BaseExpriment(BaseLogger):

    def __init__(self):
        super(BaseExpriment, self).__init__()
        self.load_config()
        # self.logger = Logger.get_logger()

        pass

    def load_config(self):
        raise NotImplementedError

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
    expriment = BaseExpriment()
    expriment.train()
    expriment.show_history()
