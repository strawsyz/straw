import os

from utils import file_utils
from utils import time_utils


class DeepConfig:
    def __init__(self):
        self.random_state = 0
        self.num_epoch = 1
        self.batch_size = 2
        self.is_use_gpu = False
        # 必须有的参数
        self.optim_name = "optim"
        self.history_save_path = "D:\Download\models\polyp"
        self.model_save_path = "D:\Download\models\polyp"
        self.model_save_path = os.path.join(self.model_save_path, time_utils.get_date())
        file_utils.make_directory(self.model_save_path)
        self.lr = 0.002
        # if None,then use all data
        self.num_train = 600
        self.num_test = 140
        self.batch_size4test = 8
        self.is_pretrain = True
        self.valid_rate = 0.5
        self.pretrain_path = "D:\Download\models\polyp\\2020-07-11\ep13_00-10-58.pkl"
        self.result_save_path = "D:\Download\models\deepeasy"
