import platform
from abc import abstractmethod

from base.base_recorder import EpochRecord
from utils.time_utils import get_time4filename


class BaseExperimentConfig:
    def __init__(self):
        self.use_prettytable = False
        self._system = platform.system()
        self.dataset_config = None
        self.net_config = None

    @abstractmethod
    def set_dataset(self):
        raise NotImplementedError

    @abstractmethod
    def set_net(self):
        raise NotImplementedError

    @abstractmethod
    def set_model_selector(self):
        raise NotImplementedError

    def __repr__(self):
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
        elif self._system == 'Linux' or (self._system == 'Windows' and self.use_prettytable == False):
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
            self.history_save_path = \
                os.path.join(self.history_save_dir, "history_{}.pth".format(get_time4filename()))
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

        self.set_net()

        self.set_dataset()
        self.set_model_selector()
        self.recorder = EpochRecord
        from base.base_recorder import ExperimentRecord
        self.experiment_record = ExperimentRecord

        if self._system == "Windows":
            self.num_epoch = 500
            self.is_use_gpu = False
            self.is_pretrain = False
            self.model_save_path = "D:\Download\models\polyp"
            self.history_save_dir = "D:\Download\models\polyp"
            self.history_save_path = "D:\Download\models\polyp\hitory_1594448749.744286.pth"
            self.pretrain_path = "D:\Download\models\polyp\ep825_07-17-43.pkl"
            self.result_save_path = "D:\Download\models\polyp\\result"
            # temp
            self.history_save_dir = "C:\data_analysis\models"
            self.history_save_path = "C:\data_analysis\models\history.pth"
            self.model_save_path = "C:\data_analysis\models\images"
            self.is_pretrain = False
            self.pretrain_path = "C:\data_analysis\models\polyp\\2020-07-11\ep0_14-50-09.pkl"
            self.result_save_path = "C:\data_analysis\models\images"
        elif self._system == "Linux":
            self.num_epoch = 1000
            self.is_use_gpu = True
            self.is_pretrain = False
            self.history_save_dir = "/home/straw/Downloads/models/polyp/history"
            # STEP 2
            # self.history_save_path = '/home/straw/Downloads/models/polyp/history/history_1596504981.pth'
            # not use edge
            # self.history_save_path = '/home/straw/Downloads/models/polyp/history/history_1596727136.pth'
            # use edge
            # self.history_save_path = '/home/straw/Downloads/models/polyp/history/history_1596849406.pth'
            # self.history_save_path = "/home/straw/Downloads/models/polyp/history/history_1596889557.pth"
            self.history_save_path = "/home/straw/Downloads/models/polyp/history/history_08-14_09-24-44.pth"
            # use new network two input one output
            # todo 覆盖保存历史记录的时候也许需要提醒一下
            # self.history_save_path = "/home/straw/Downloads/models/polyp/history/history1595211050.pth"
            # self.history_save_path = "/home/straw/Downloads/models/polyp/history/history_1596127749.pth"
            # self.pretrain_path = "/home/straw/Downloads/models/polyp/2020-07-18/ep212_20-10-46.pkl"
            # self.pretrain_path = '/home/straw/Downloads/models/polyp/2020-07-18/ep331_22-49-39.pkl'
            # self.pretrain_path = '/home/straw/Downloads/models/polyp/2020-07-23/ep638_16-46-04.pkl'
            # best model in step one
            # self.pretrain_path = '/home/straw/Downloads/models/polyp/2020-08-01/ep410_00-30-36.pkl'
            # step two
            # self.pretrain_path = '/home/straw/Downloads/models/polyp/2020-08-03/ep395_14-37-02.pkl'
            self.pretrain_path = '/home/straw/Downloads/models/polyp/2020-08-05/ep241_00-36-43.pkl'
            # not use edge
            self.pretrain_path = '/home/straw/Downloads/models/polyp/2020-08-07/ep68_03-13-11.pkl'
            # use edge
            self.pretrain_path = '/home/straw/Downloads/models/polyp/2020-08-08/ep82_13-44-27.pkl'
            self.pretrain_path = '/home/straw/Downloads/models/polyp/2020-08-09/ep61_00-00-39.pkl'
            # INFO]<2020-08-07 03:12:55,755> { EPOCH:68	 train_loss:0.284118 }
            # [INFO]<2020-08-07 03:13:11,073> { Epoch:68	 valid_loss:0.345331 }
            # [INFO]<2020-08-07 03:13:11,073> { ========== saving history========== }
            # [INFO]<2020-08-07 03:13:11,080> { ========== saved history at /home/straw/Downloads/models/polyp/history/history_1596727136.pth========== }
            # [INFO]<2020-08-07 03:13:11,081> { save this model for ['train_loss', 'valid_loss'] is better }
            # [INFO]<2020-08-07 03:13:11,081> { ==============saving model data=============== }
            # [INFO]<2020-08-07 03:13:11,256> { ==============saved at /home/straw/Downloads/models/polyp/2020-08-07/ep68_03-13-11.pkl=============== }
            # [INFO]<2020-08-07 03:13:11,256> { use 159 seconds in the epoch }
            self.model_save_path = "/home/straw/Downloads/models/polyp/"
            self.result_save_path = "/home/straw/Download\models\polyp\\result"
        # remove some useless or wrong attribute
        self.init_attr()

    def set_net(self):
        # STEP 1 use source image to predict mask image
        # from configs.net_config import FCNNetConfig
        # self.net_config = FCNNetConfig()
        # self.net = FCNNet(self.net_config)

        # STEP 2 USE predict result in step 1 and edge image to predict image
        from configs.net_config import FCNNet4EdgeConfig
        self.net_config = FCNNet4EdgeConfig()
        from nets import FCN4EdgeNet
        self.net = FCN4EdgeNet(self.net_config)

    def set_dataset(self):
        # dataset setting
        # from configs.dataset_config import ImageDataSetConfig
        # self.dataset_config = ImageDataSetConfig()
        # self.dataset = ImageDataSet(self.dataset_config)

        # from configs.dataset_config import ImageDataSet4TestConfig
        # self.dataset_config = ImageDataSet4TestConfig()
        # from datasets.image_dataset import ImageDataSet4Test
        # self.dataset = ImageDataSet4Test(self.dataset_config)

        from configs.dataset_config import ImageDataSet4EdgeConfig
        self.dataset_config = ImageDataSet4EdgeConfig()
        from datasets.image_dataset import ImageDataSet4Edge
        self.dataset = ImageDataSet4Edge(self.dataset_config)

    def set_model_selector(self):
        from base.base_model_selector import BaseSelector, ScoreModel
        score_models = []
        score_models.append(ScoreModel("train_loss", bigger_better=False))
        score_models.append(ScoreModel("valid_loss", bigger_better=False))
        self.model_selector = BaseSelector(score_models)


class FNNConfig(BaseExperimentConfig):

    def __init__(self):
        super(FNNConfig, self).__init__()
        self.recorder = EpochRecord
        self.set_net()
        self.set_dataset()
        self.set_model_selector()
        self.num_epoch = 500
        self.recorder = EpochRecord
        from base.base_recorder import ExperimentRecord
        self.experiment_record = ExperimentRecord
        # self.history_save_dir = "D:\models\Ecg"
        # self.history_save_path = ""
        # self.model_save_path = "D:\models\Ecg"
        # self.result_save_path = "D:\models\Ecg\results"

        self.history_save_dir = "C:\data_analysis\models"
        # self.history_save_path = "C:\data_analysis\models\history1595256151.pth"
        self.history_save_path = "C:\data_analysis\models\history_08-08_15-05-21.pth"
        self.model_save_path = "C:\data_analysis\models\deepeasy"
        self.result_save_path = "C:\data_analysis\models\deepeasy"

        # if None,then use all data
        # self.is_pretrain = False
        self.is_pretrain = True
        # self.pretrain_path = "C:\data_analysis\models\deepeasy\2020-08-07\ep176_21-37-20.pkl"
        self.pretrain_path = "C:\data_analysis\models\deepeasy\\2020-08-08\ep4_15-05-49.pkl"
        # self.pretrain_path = "C:\data_analysis\models\history_2020-08-07_20-55-38.pth"
        # "D:\models\Ecg\2020-07-22\ep48_11-53-23.pkl"

        # is use prettytable
        self.use_prettytable = False

        self.init_attr()

    def set_net(self):
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

    def set_dataset(self):
        # dataset
        from configs.dataset_config import CSVDataSetConfig
        self.dataset_config = CSVDataSetConfig()
        from datasets import CsvDataSet
        self.dataset = CsvDataSet(self.dataset_config)

    def set_model_selector(self):
        # model selector
        from base.base_model_selector import BaseSelector
        self.model_selector = BaseSelector()
        self.model_selector.regist_score_model("train_loss", bigger_better=False)
        self.model_selector.regist_score_model("valid_loss", bigger_better=False)
