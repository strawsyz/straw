from base.base_recorder import BaseHistory
from datasets.csv_dataset import CsvDataSet
from datasets.image_dataset import ImageDataSet
from nets.FCNNet import FCNNet
from nets.FNNNet import MyFNN


class BaseExperimentConfig:
    def __repr__(self):
        import platform
        delimiter = "↓" * 50
        if platform.system() == 'Linux':
            config_items = []
            config_items.append("dataset config\t" + "↓" * 50)

            for attr in self.dataset_config.__dict__:
                config_items.append("{}\t{}".format(attr, getattr(self.dataset_config, attr)))
            for attr in self.net_config.__dict__:
                config_items.append("{}\t{}".format(attr, getattr(self.net_config, attr)))
            for attr in self.__dict__:
                config_items.append("{}\t{}".format(attr, getattr(self, attr)))
            return "\n".join(config_items)
        else:
            from prettytable import PrettyTable
            config_view = PrettyTable()
            config_view.field_names = ["name", "value"]
            # todo 可能之后改为__slots__
            config_view.add_row(["dataset config", delimiter])
            for attr in self.dataset_config.__dict__:
                config_view.add_row([attr, getattr(self.dataset_config, attr)])
            config_view.add_row(["network config", delimiter])
            for attr in self.net_config.__dict__:
                config_view.add_row([attr, getattr(self.net_config, attr)])
            config_view.add_row(["experiment config", delimiter])
            for attr in self.__dict__:
                config_view.add_row([attr, getattr(self, attr)])
            return "\n{}".format(str(config_view))

    def init_attr(self):
        import os
        if not hasattr(self, "history_save_path") or \
                not isinstance(self.history_save_path, str) or \
                not os.path.isfile(self.history_save_path):
            import time
            self.history_save_path = \
                os.path.join(self.history_save_dir, "history_{}.pth".format(int(time.time())))
        from utils.time_utils import get_date
        self.result_save_path = os.path.join(self.result_save_path, get_date())


class ImageSegmentationConfig(BaseExperimentConfig):
    def __init__(self):
        self.recorder = BaseHistory
        from configs.net_config import FCNNetConfig
        self.net_config = FCNNetConfig()
        self.net = FCNNet(self.net_config)
        from configs.dataset_config import ImageDataSetConfig
        self.dataset_config = ImageDataSetConfig()
        self.dataset = ImageDataSet(self.dataset_config)
        from base.base_model_selector import BaseSelector, ScoreModel
        score_models = []
        score_models.append(ScoreModel("train_loss", bigger_better=False))
        score_models.append(ScoreModel("valid_loss", bigger_better=False))
        self.model_selector = BaseSelector(score_models)
        self.num_epoch = 500

        # temp
        # self.history_save_dir = "C:\models"
        # self.history_save_path = "C:\models\history.pth"
        # self.model_save_path = "C:\models\deepeasy"
        # self.is_pretrain = False
        # self.pretrain_path = "C:\models\polyp\\2020-07-11\ep0_14-50-09.pkl"
        # self.result_save_path = "C:\models\deepeasy"

        import platform
        __system = platform.system()

        if __system == "Windows":
            self.is_use_gpu = False
            self.is_pretrain = False
            self.model_save_path = "D:\Download\models\polyp"
            self.history_save_dir = "D:\Download\models\polyp"
            self.history_save_path = "D:\Download\models\polyp\hitory_1594448749.744286.pth"
            self.pretrain_path = "D:\Download\models\polyp\ep825_07-17-43.pkl"
            self.result_save_path = "D:\Download\models\polyp\\result"
        elif __system == "Linux":
            self.num_epoch = 1000
            self.is_use_gpu = True
            self.is_pretrain = True
            self.history_save_dir = "/home/straw/Downloads/models/polyp/history"
            # todo 覆盖保存历史记录的时候也许需要提醒一下
            self.history_save_path = None
            # self.history_save_path = ""
            # self.pretrain_path = "/home/straw/Downloads/models/polyp/2020-07-18/ep212_20-10-46.pkl"
            # self.pretrain_path = '/home/straw/Downloads/models/polyp/2020-07-18/ep331_22-49-39.pkl'
            self.pretrain_path = '/home/straw/Downloads/models/polyp/2020-07-19/ep825_07-17-43.pkl'
            self.model_save_path = "/home/straw/Downloads/models/polyp/"
            self.result_save_path = "/home/straw/Download\models\polyp\\result"

        self.init_attr()


class FNNConfig(BaseExperimentConfig):
    def __init__(self):
        self.recorder = BaseHistory
        from configs.net_config import FNNNetConfig
        self.net_config = FNNNetConfig()
        from nets.FNNNet import FNNNet
        self.net = FNNNet(self.net_config)
        from configs.dataset_config import CSVDataSetConfig
        self.dataset_config = CSVDataSetConfig()
        from datasets.csv_dataset import CsvDataSet
        self.dataset = CsvDataSet(self.dataset_config)
        from base.base_model_selector import BaseSelector, ScoreModel
        # score_models = []
        # score_models.append(ScoreModel("train_loss", bigger_better=False))
        # score_models.append(ScoreModel("valid_loss", bigger_better=False))
        self.model_selector = BaseSelector()
        self.model_selector.add_score("train_loss", bigger_better=False)
        self.model_selector.add_score("valid_loss", bigger_better=False)
        # self.model_selector = selector
        self.num_epoch = 50
        self.recorder = BaseHistory

        # temp
        self.history_save_dir = "C:\data_analysis\models"
        self.history_save_path = "C:\data_analysis\models\history1595256151.pth"
        self.model_save_path = "C:\data_analysis\models\deepeasy"
        # if None,then use all data
        self.is_pretrain = True
        self.pretrain_path = "C:\data_analysis\models\deepeasy\\2020-07-20\ep45_23-45-16.pkl"
        self.result_save_path = "C:\data_analysis\models\deepeasy"
        self.init_attr()
