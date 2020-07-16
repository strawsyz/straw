from nets.FNNNet import MyFNN
from base.base_recorder import BaseHistory
from datasets.image_dataset import ImageDataSet
from datasets.csv_dataset import CsvDataSet
from nets.FCNNet import FCNNet


class ImageSegmentationConfig:
    def __init__(self):
        self.recorder = BaseHistory
        self.net = FCNNet
        self.dataset = ImageDataSet

        self.num_epoch = 5
        self.is_use_gpu = False
        self.history_save_dir = "D:\Download\models\polyp"
        self.history_save_path = "D:\Download\models\polyp\hitory_1594448749.744286.pth"
        self.model_save_path = "D:\Download\models\polyp"

        self.is_pretrain = True
        self.pretrain_path = "D:\Download\models\polyp\\2020-07-11\ep0_14-50-09.pkl"
        self.result_save_path = "D:\Download\models\deepeasy"

        # temp
        self.history_save_dir = "C:\models"
        self.history_save_path = "C:\models\history.pth"
        self.model_save_path = "C:\models\deepeasy"
        self.is_pretrain = False
        self.pretrain_path = "C:\models\polyp\\2020-07-11\ep0_14-50-09.pkl"
        self.result_save_path = "C:\models\deepeasy"


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
