from base.base_recorder import BaseHistory
from datasets import ImageDataSet
from nets import FCNNet
import platform


class BaseExperimentConfig:
    def __init__(self):
        self.use_prettytable = False
        self._system = platform.system()

    def __repr__(self):
        import platform
        delimiter = "↓" * 50
        if self.use_prettytable and self._system == "Windows":
            from prettytable import PrettyTable
            config_view = PrettyTable()
            config_view.field_names = ["name", "value"]
            # todo 可能之后改为__slots__
            config_view.add_row(["dataset config", delimiter])
            for attr in sorted(self.dataset_config.__dict__):
                config_view.add_row([attr, getattr(self.dataset_config, attr)])
            config_view.add_row(["network config", delimiter])
            for attr in sorted(self.net_config.__dict__):
                config_view.add_row([attr, getattr(self.net_config, attr)])
            config_view.add_row(["experiment config", delimiter])
            for attr in sorted(self.__dict__):
                config_view.add_row([attr, getattr(self, attr)])
            return "\n{}".format(str(config_view))
        elif self._system == 'Linux':
            config_items = []
            config_items.append("dataset config\t" + delimiter)
            for attr in sorted(self.dataset_config.__dict__):
                config_items.append("{}\t{}".format(attr, getattr(self.dataset_config, attr)))
            config_items.append("network config\t" + delimiter)
            for attr in sorted(self.net_config.__dict__):
                config_items.append("{}\t{}".format(attr, getattr(self.net_config, attr)))
            config_items.append("experiment config\t" + delimiter)
            for attr in sorted(self.__dict__):
                config_items.append("{}\t{}".format(attr, getattr(self, attr)))
            return "\n".join(config_items)

    def init_attr(self):
        import os
        if not hasattr(self, "history_save_path") or \
                not isinstance(self.history_save_path, str) or \
                not os.path.isfile(self.history_save_path):
            import time
            self.history_save_path = \
                os.path.join(self.history_save_dir, "history_{}.pth".format(int(time.time())))
        from utils.time_utils import get_date
        if getattr(self, "history_save_path", None):
            self.result_save_path = os.path.join(self.result_save_path, get_date())

        for attr_name in self.__dict__:
            attr_value = getattr(self, attr_name)
            if isinstance(attr_value, str) and attr_value.split() == "":
                setattr(self, attr_name, None)


class ImageSegmentationConfig(BaseExperimentConfig):
    def __init__(self):
        super(ImageSegmentationConfig, self).__init__()
        self.use_prettytable = True

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


        if self._system == "Windows":
            self.is_use_gpu = False
            self.is_pretrain = False
            self.model_save_path = "D:\Download\models\polyp"
            self.history_save_dir = "D:\Download\models\polyp"
            self.history_save_path = "D:\Download\models\polyp\hitory_1594448749.744286.pth"
            self.pretrain_path = "D:\Download\models\polyp\ep825_07-17-43.pkl"
            self.result_save_path = "D:\Download\models\polyp\\result"
            # temp
            # self.history_save_dir = "C:\models"
            # self.history_save_path = "C:\models\history.pth"
            # self.model_save_path = "C:\models\deepeasy"
            # self.is_pretrain = False
            # self.pretrain_path = "C:\models\polyp\\2020-07-11\ep0_14-50-09.pkl"
            # self.result_save_path = "C:\models\deepeasy"
        elif self._system == "Linux":
            self.num_epoch = 2
            self.is_use_gpu = True
            self.is_pretrain = True
            self.history_save_dir = "/home/straw/Downloads/models/polyp/history"
            # todo 覆盖保存历史记录的时候也许需要提醒一下
            self.history_save_path = "/home/straw/Downloads/models/polyp/history/history1595211050.pth"
            # self.history_save_path = ""
            # self.pretrain_path = "/home/straw/Downloads/models/polyp/2020-07-18/ep212_20-10-46.pkl"
            # self.pretrain_path = '/home/straw/Downloads/models/polyp/2020-07-18/ep331_22-49-39.pkl'
            self.pretrain_path = '/home/straw/Downloads/models/polyp/2020-07-23/ep638_16-46-04.pkl'
            self.model_save_path = "/home/straw/Downloads/models/polyp/"
            self.result_save_path = "/home/straw/Download\models\polyp\\result"

        self.init_attr()


class FNNConfig(BaseExperimentConfig):

    def __init__(self):
        super(FNNConfig, self).__init__()
        self.recorder = BaseHistory
        self.net_name = "fnn"
        # net
        if self.net_name == "cnn1d":
            from configs.net_config import CNN1DNetConfig
            self.net_config = CNN1DNetConfig()
            from nets.CNN1DNet import CNN1DNet
            self.net = CNN1DNet(self.net_config)
        elif self.net_name == "fnn":
            from configs.net_config import FNNNetConfig
            self.net_config = FNNNetConfig()
            from nets import FNNNet
            self.net = FNNNet(self.net_config)
        # dataset
        from configs.dataset_config import CSVDataSetConfig
        self.dataset_config = CSVDataSetConfig()
        from datasets import CsvDataSet
        # from datasets.csv_dataset import CsvDataSet
        self.dataset = CsvDataSet(self.dataset_config)
        # model selector
        from base.base_model_selector import BaseSelector
        self.model_selector = BaseSelector()
        self.model_selector.add_score("train_loss", bigger_better=False)
        self.model_selector.add_score("valid_loss", bigger_better=False)

        self.num_epoch = 500
        self.recorder = BaseHistory

        # self.history_save_dir = "D:\models\Ecg"
        # self.history_save_path = ""
        # self.model_save_path = "D:\models\Ecg"
        # self.result_save_path = "D:\models\Ecg\results"

        self.history_save_dir = "C:\data_analysis\models"
        self.history_save_path = "C:\data_analysis\models\history1595256151.pth"
        self.model_save_path = "C:\data_analysis\models\deepeasy"
        self.result_save_path = "C:\data_analysis\models\deepeasy"

        # if None,then use all data
        self.is_pretrain = True
        # self.is_pretrain = True
        # self.pretrain_path = "C:\data_analysis\models\deepeasy\\2020-07-20\ep45_23-45-16.pkl"
        # self.pretrain_path = "C:\data_analysis\models\deepeasy\\2020-07-23\ep31_19-59-45.pkl"
        self.pretrain_path = "C:\data_analysis\models\deepeasy\\2020-07-24\ep1_09-46-26.pkl"
        # "D:\models\Ecg\2020-07-22\ep48_11-53-23.pkl"

        # is use prettytable
        self.use_prettytable = True

        self.init_attr()
