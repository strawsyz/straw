from base.base_recorder import BaseHistory
from datasets.csv_dataset import CsvDataSet
from datasets.image_dataset import ImageDataSet
from nets.FCNNet import FCNNet
from nets.FNNNet import MyFNN


class BaseExperimentConfig:
    def __repr__(self):
        from prettytable import PrettyTable
        config_view = PrettyTable()
        config_view.field_names = ["name", "value"]
        # todo 可能之后改为__slots__
        config_view.add_row(["dataset config", "↓" * 10])
        for attr in self.dataset_config.__dict__:
            config_view.add_row([attr, getattr(self.dataset_config, attr)])
        config_view.add_row(["network config", "↓" * 10])
        for attr in self.net_config.__dict__:
            config_view.add_row([attr, getattr(self.net_config, attr)])
        config_view.add_row(["experiment config", "↓" * 50])
        for attr in self.__dict__:
            config_view.add_row([attr, getattr(self, attr)])
        # return config_view
        return "\n{}".format(config_view)


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
        self.selector = BaseSelector(score_models)
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
            self.history_save_dir = "D:\Download\models\polyp"
            self.history_save_path = "D:\Download\models\polyp\hitory_1594448749.744286.pth"
            self.pretrain_path = "D:\Download\models\polyp\FCN_NLL_ep1989_04-57-10.pkl"
            self.model_save_path = "D:\Download\models\polyp"
            self.result_save_path = "D:\Download\models\polyp\\result"

        elif __system == "Linux":
            self.num_epoch = 1000
            self.is_use_gpu = True
            self.is_pretrain = True
            self.history_save_dir = "/home/straw/Downloads/models/polyp/history"
            self.history_save_path = "/home/straw/Downloads/models/polyp/history/history.pth"
            # self.pretrain_path = "/home/straw/Downloads/models/polyp/2020-07-18/ep212_20-10-46.pkl"
            self.pretrain_path = '/home/straw/Downloads/models/polyp/2020-07-18/ep331_22-49-39.pkl'
            self.model_save_path = "/home/straw/Downloads/models/polyp/"
            self.result_save_path = "/home/straw/Download\models\polyp\\result"


class FNNConfig:
    def __init__(self):
        self.recorder = BaseHistory
        self.net = MyFNN
        self.dataset = CsvDataSet

        # temp
        self.history_save_dir = "C:\models"
        self.history_save_path = "C:\models\history.pth"
        self.model_save_path = "C:\models\deepeasy"
        # if None,then use all data
        self.is_pretrain = False
        self.pretrain_path = ""
        self.result_save_path = "C:\models\deepeasy"
