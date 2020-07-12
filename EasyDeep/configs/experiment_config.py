from base.base_recorder import HistoryRecorder
from datasets.image_dataset import ImageDataSet
from nets.FCNNet import FCNNet


class DeepConfig:
    def __init__(self):
        self.recorder = HistoryRecorder
        self.net = FCNNet
        self.dataset = ImageDataSet

        self.num_epoch = 2
        self.is_use_gpu = False
        self.history_save_dir = "D:\Download\models\polyp"
        self.history_save_path = "D:\Download\models\polyp\hitory_1594448749.744286.pth"
        self.model_save_path = "D:\Download\models\polyp"
        # if None,then use all data

        self.is_pretrain = True
        self.pretrain_path = "D:\Download\models\polyp\\2020-07-11\ep0_14-50-09.pkl"
        self.result_save_path = "D:\Download\models\deepeasy"
