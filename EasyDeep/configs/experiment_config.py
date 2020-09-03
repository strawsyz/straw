import os
from utils.time_utils import get_date
import platform
from abc import ABC, abstractmethod

from base.base_recorder import EpochRecord
from utils.time_utils import get_time4filename
from base.base_logger import BaseLogger


class BaseExperimentConfig(ABC, BaseLogger):
    def __init__(self, tag=None):
        self.use_prettytable = False
        self._system = platform.system()
        self.dataset_config = None
        self.net_config = None
        self.setter(tag=tag)
        self.tag = tag
        self.history_save_path = None

    @abstractmethod
    def set_dataset(self):
        pass

    @abstractmethod
    def set_net(self):
        pass

    @abstractmethod
    def set_model_selector(self):
        pass

    def __repr__(self):
        delimiter = "↓" * 50
        if self.use_prettytable and self._system == "Windows":
            from prettytable import PrettyTable
            config_view = PrettyTable()
            config_view.field_names = ["name", "value"]
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
        if not isinstance(self.history_save_path, str) or \
                not os.path.isfile(self.history_save_path):
            self.history_save_path = \
                os.path.join(self.history_save_dir, "{}_{}.pth".format(self.tag, get_time4filename()))
        if getattr(self, "history_save_path", None):
            self.result_save_path = os.path.join(self.result_save_path, get_date())

        for attr_name in self.__dict__:
            attr_value = getattr(self, attr_name)
            if isinstance(attr_value, str) and attr_value.split() == "":
                setattr(self, attr_name, None)

    def setter(self, attr_names=["net", "dataset"], tag=None):
        if tag is None:
            for attr_name in attr_names:
                getattr(self, "set_{}".format(attr_name))()
        else:
            for attr_name in attr_names:
                set_attr = getattr(self, "set_{}_{}".format(attr_name, tag), None)
                if set_attr is None:
                    getattr(self, "set_{}".format(attr_name))()
                else:
                    set_attr()


class ImageSegmentationConfig(BaseExperimentConfig):
    def __init__(self, tag=None):
        super(ImageSegmentationConfig, self).__init__(tag)
        self.use_prettytable = True
        self.set_model_selector()
        self.recorder = EpochRecord
        from base.base_recorder import ExperimentRecord
        self.experiment_record = ExperimentRecord

        if self._system == "Windows":
            self.num_epoch = 500
            self.is_use_gpu = True
            self.is_pretrain = True
            self.model_save_path = "C:\(lab\models\polyp"
            self.history_save_dir = "C:\(lab\models\polyp\hisotry"
            self.history_save_path = None
            # step 1
            self.pretrain_path = "C:\(lab\models\polyp\ep410_00-30-36.pkl"
            # step 2 not use edge
            self.pretrain_path = "C:\(lab\models\polyp\ep68_03-13-11.pkl"
            # step 2 use edge
            self.pretrain_path = "C:\(lab\models\polyp\ep77_13-22-08.pkl"
            self.result_save_path = "C:\(lab\models\polyp\\result"
        elif self._system == "Linux":
            self.num_epoch = 1000
            self.is_use_gpu = True
            self.is_pretrain = True
            self.history_save_dir = "/home/straw/Downloads/models/polyp/history"
            self.pretrain_path = "/home/straw/Downloads/models/polyp/2020-08-11/ep77_13-22-08.pkl"
            self.model_save_path = "/home/straw/Downloads/models/polyp/"
            self.result_save_path = "/home/straw/Download\models\polyp\\result"
        self.init_attr()

    def set_net(self):
        # STEP 1 use source image to predict mask image
        from nets.FCNNet import FCNNet
        from configs.net_config import FCNNetConfig
        self.net_config = FCNNetConfig()
        self.net = FCNNet()

    def set_net_edge(self):
        # STEP 2 USE predict result in step 1 and edge image to predict image
        from configs.net_config import FCNNet4EdgeConfig
        self.net_config = FCNNet4EdgeConfig()
        from nets import FCN4EdgeNet
        self.net = FCN4EdgeNet()

    def set_dataset(self):
        from datasets.image_dataset import ImageDataSet
        from configs.dataset_config import ImageDataSetConfig
        self.dataset_config = ImageDataSetConfig()
        self.dataset = ImageDataSet(self.dataset_config)

    def set_dataset_edge(self):
        from configs.dataset_config import ImageDataSet4EdgeConfig
        self.dataset_config = ImageDataSet4EdgeConfig()
        from datasets.image_dataset import ImageDataSet4Edge
        self.dataset = ImageDataSet4Edge(self.dataset_config)

    def set_model_selector(self):
        from base.base_model_selector import BaseModelSelector, ScoreModel
        score_models = []
        score_models.append(ScoreModel("train_loss", bigger_better=False))
        score_models.append(ScoreModel("valid_loss", bigger_better=False))
        self.model_selector = BaseModelSelector(score_models)


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

        self.history_save_dir = ""
        self.model_save_path = ""
        self.result_save_path = ""

        self.is_pretrain = False
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
        from configs.dataset_config import CSVDataSetConfig
        self.dataset_config = CSVDataSetConfig()
        # todo need set  a dataset

    def set_model_selector(self):
        from base.base_model_selector import BaseModelSelector
        self.model_selector = BaseModelSelector()
        self.model_selector.regist_score_model("train_loss", bigger_better=False)
        self.model_selector.regist_score_model("valid_loss", bigger_better=False)


class MnistConfig(BaseExperimentConfig):

    def __init__(self):
        super(MnistConfig, self).__init__()
        self.recorder = EpochRecord
        self.set_net()
        self.set_dataset()
        self.set_model_selector()
        self.current_epoch = 0
        self.num_epoch = 500
        self.recorder = EpochRecord
        from base.base_recorder import ExperimentRecord
        self.experiment_record = ExperimentRecord

        self.history_save_dir = ""
        self.model_save_path = ""
        self.result_save_path = ""

        self.is_pretrain = False
        self.is_use_gpu = False
        self.use_prettytable = False

        self.init_attr()

    def set_net(self):
        from nets.MnistNet import MnistNet
        from configs.net_config import MnistConfig
        self.net_config = MnistConfig()
        self.net = MnistNet()

    def set_dataset(self):
        from configs.dataset_config import CSVDataSetConfig
        self.dataset_config = CSVDataSetConfig()
        from datasets.mnist_dataset import MNISTDataset
        self.dataset = MNISTDataset()

    def set_model_selector(self):
        from base.base_model_selector import BaseModelSelector
        self.model_selector = BaseModelSelector()
        self.model_selector.regist_score_model("train_loss", bigger_better=False)
        self.model_selector.regist_score_model("valid_loss", bigger_better=False)
