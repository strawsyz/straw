import torch


class EpochRecord:
    """记录一个epoch的训练结果"""

    def __init__(self, train_loss, valid_loss=None):
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.scores = ["train_loss", "valid_loss"]

    def get_record_dict(self):
        record_dict = {}
        for attr in self.scores:
            record_dict[attr] = getattr(self, attr, None)
        return record_dict

    def __lr__(self):
        return True


class ExperimentRecord:
    def __init__(self):
        # key:epoch, value:EpochRecord
        self.epoch_records = {}
        # 保存配置的信息
        self.config_info = None
        # every score_model have a best_model_path
        self.best_model_paths = {}

    def add_scores(self, epoch, epoch_record):
        self.epoch_records[epoch] = epoch_record

    def save(self, save_path):
        torch.save(self, save_path)

    def load(self, save_path):
        # import pickle
        # with open(save_path, mode="rb+") as f:
        #     obj = pickle.load(f)
        #     print(dir(obj))
        #     for key in obj:
        #         print("{}:{}".format(key, obj[key]))
        #     return obj

        return torch.load(save_path)


class CustomEpochRecord(EpochRecord):
    def __init__(self, attr_names: list = []):
        for attr_name in attr_names:
            setattr(self, attr_name, None)

    def __call__(self, *args, **kwargs):
        for attr_name in kwargs:
            setattr(self, attr_name, kwargs.get(attr_name))

    def __str__(self):
        return str(self.get_record_dict())

    def merge_record(self, record):
        """
        合并两个record
        如果新设置的record有重复的部分，会覆盖到之前的数据上
        :param record:
        :return:
        """
        for attr_name in record.__dict__:
            setattr(self, attr_name, getattr(record, attr_name))
        return self

    def set_attr(self, attr_name, attr_value):
        """
        设置参数
        :param attr_name:
        :param attr_value:
        :return:
        """
        if attr_name in self.__dict__:
            setattr(self, attr_name, attr_value)
        else:
            raise RuntimeError("can't set a attribute which is not exist")

    def get_record_dict(self):
        record_dict = {}
        for attr in self.__dict__:
            record_dict[attr] = getattr(self, attr, None)
        return record_dict
