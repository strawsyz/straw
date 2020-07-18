from base.base_recorder import BaseHistory
from datasets.csv_dataset import CsvDataSet
from datasets.image_dataset import ImageDataSet
from nets.FCNNet import FCNNet
from nets.FNNNet import MyFNN


class ImageSegmentationConfig:
    def __init__(self):
        self.recorder = BaseHistory
        self.net = FCNNet
        self.dataset = ImageDataSet

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
