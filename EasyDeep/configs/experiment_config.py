from base.base_recorder import ExperimentRecord
import os

from base.base_config import BaseExperimentConfig
from base.base_recorder import EpochRecord
from utils.time_utils import get_date
from utils.time_utils import get_time_4_filename


class DeepExperimentConfig(BaseExperimentConfig):
    def __init__(self):
        super(DeepExperimentConfig, self).__init__()
        self.other_param_4_batch = None

        self.current_epoch = 0
        self.scores_history = {}
        self.recorder = None

        self.num_epoch = None
        self.is_use_gpu = None
        self.history_save_dir = None
        self.history_save_path = None
        self.model_save_path = None
        self.model_selector = None
        self.is_pretrain = None
        self.pretrain_path = None
        self.result_save_path = None
        self.selector = None
        self.is_bigger_better = False
        self.num_iter = 50

        self.dataset = None
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None

        self.net = None
        self.net_structure = None
        self.optimizer = None
        self.loss_function = None
        self.scheduler = None

        self.setter_by_tag(tag=self.tag)

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
            if hasattr(self, "net_config") and self.net_config is not None:
                net_config = self.net_config
            else:
                net_config = self.net
            for attr in sorted(net_config.__dict__):
                config_view.add_row([attr, getattr(net_config, attr)])
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
                os.path.join(self.history_save_dir, "{}_{}.pth".format(self.tag, get_time_4_filename()))
        if getattr(self, "history_save_path", None):
            self.result_save_path = os.path.join(self.result_save_path, get_date())

        for attr_name in self.__dict__:
            attr_value = getattr(self, attr_name)
            if isinstance(attr_value, str) and attr_value.split() == "":
                setattr(self, attr_name, None)

    def setter_by_tag(self, attr_names=["net", "dataset"], tag=None):
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

    def show_config(self):
        """list all configure in a experiment"""
        self.config_info = str(self)
        self.logger.info(self.config_info)
        return self.config_info


class ImageSegmentationConfig(DeepExperimentConfig):
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
            self.is_pretrain = False
            self.model_save_path = r"C:\(lab\models\polyp"
            self.history_save_dir = r"C:\(lab\models\polyp\hisotry"
            self.history_save_path = None
            # step 1
            self.pretrain_path = r"C:\(lab\models\polyp\ep410_00-30-36.pkl"
            # step 2 not use edge
            self.pretrain_path = r"C:\(lab\models\polyp\ep68_03-13-11.pkl"
            # step 2 use edge
            self.pretrain_path = r"C:\(lab\models\polyp\ep77_13-22-08.pkl"
            self.result_save_path = r"C:\(lab\models\polyp\\result"
        elif self._system == "Linux":
            self.num_epoch = 1000
            self.is_use_gpu = True
            self.is_pretrain = True
            self.history_save_dir = r"/home/straw/Downloads/models/polyp/history"
            self.pretrain_path = r"/home/straw/Downloads/models/polyp/2020-08-11/ep77_13-22-08.pkl"
            self.model_save_path = r"/home/straw/Downloads/models/polyp/"
            self.result_save_path = r"/home/straw/Download\models\polyp/result"
        self.init_attr()

    def set_net(self):
        # STEP 1 use source image to predict mask image
        # from nets.FCNNet import FCNNet
        # from configs.net_config import FCNNetConfig
        # self.net_config = FCNNetConfig()
        # self.net = FCNNet()
        # replace vgg16 by resnet
        from nets.FCNNet import FCNResNet
        self.net = FCNResNet()

    def set_net_edge(self):
        # STEP 2 USE predict result in step 1 and edge image to predict image
        from configs.net_config import FCNBaseNet4EdgeConfig
        self.net_config = FCNBaseNet4EdgeConfig()
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


class FNNConfig(DeepExperimentConfig):

    def __init__(self):
        super(FNNConfig, self).__init__()
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
            from configs.net_config import CNN1DBaseNetConfig
            self.net_config = CNN1DBaseNetConfig()
            from nets.CNN1DNet import CNN1DNet
            self.net = CNN1DNet(self.net_config)
        elif self.net_name == "fnn":
            from configs.net_config import FNNBaseNetConfig
            self.net_config = FNNBaseNetConfig()
            from nets import FNNNet
            self.net = FNNNet()

    def set_dataset(self):
        from configs.dataset_config import CSVDataSetConfig
        self.dataset_config = CSVDataSetConfig()
        # todo need set  a dataset

    def set_model_selector(self):
        from base.base_model_selector import BaseModelSelector
        self.model_selector = BaseModelSelector()
        self.model_selector.register_score_model("train_loss", bigger_better=False)
        self.model_selector.register_score_model("valid_loss", bigger_better=False)


class MnistConfig(DeepExperimentConfig):
    name = "mnist"
    tag = None

    def __init__(self):
        super(MnistConfig, self).__init__()
        self.set_net()
        self.set_dataset()
        self.set_model_selector()
        self.current_epoch = 0
        self.num_epoch = 5
        self.recorder = EpochRecord
        self.experiment_record = ExperimentRecord
        self.num_iter = 2

        self.root_path = os.path.join(r"C:\(lab\models\mnist", self.create_experiment_name(MnistConfig.name))
        self.history_save_dir = os.path.join(self.root_path, "history")
        self.model_save_path = os.path.join(self.root_path, "model")
        self.result_save_path = os.path.join(self.root_path, "result")

        self.is_pretrain = False
        self.is_use_gpu = False
        self.use_prettytable = True

        self.init_attr()
        self.show_config()

    def set_net(self):
        from nets.MnistNet import MnistNet
        from configs.net_config import MnistConfigBase
        self.net_config = MnistConfigBase()
        self.net = MnistNet()

    def set_dataset(self):
        from configs.dataset_config import CSVDataSetConfig
        self.dataset_config = CSVDataSetConfig()
        from datasets.mnist_dataset import MNISTDataset
        self.dataset = MNISTDataset()

    def set_model_selector(self):
        from base.base_model_selector import BaseModelSelector
        self.model_selector = BaseModelSelector()
        self.model_selector.register_score_model("train_loss", bigger_better=False)
        self.model_selector.register_score_model("valid_loss", bigger_better=False)
