import torch


class EpochRecord:
    """record the result of one epoch"""

    def __init__(self, train_loss, valid_loss=None):
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.model_path = None
        # the name of score used to estimate network
        self.scores = ["train_loss", "valid_loss"]

    def record_2_dict(self):
        record_dict = {}
        for attr in self.scores:
            record_dict[attr] = getattr(self, attr, None)
        return record_dict

    def __str__(self):
        return "EpochRecord"

    @classmethod
    def get_class_name(cls):
        return __class__.__name__


class ExperimentRecord:

    def __init__(self):
        # key:epoch, value:EpochRecord
        self.epoch_records = {}
        # information of the experiment's configure
        self.config_info = None
        # every score_model have a best_model_path
        self.best_model_paths = {}

    def add_scores(self, epoch, epoch_record):
        self.epoch_records[epoch] = epoch_record

    def save(self, save_path):
        torch.save(self, save_path)

    def load(self, save_path):
        return torch.load(save_path)

    def __str__(self):
        return "ExperimentRecord"

    @classmethod
    def get_class_name(cls):
        return __class__.__name__


class CustomEpochRecord(EpochRecord):

    def __init__(self, attr_names: list):
        for attr_name in attr_names:
            setattr(self, attr_name, None)
        self.attr_names = attr_names

    def __call__(self, *args, **kwargs):
        for attr_name in kwargs:
            setattr(self, attr_name, kwargs.get(attr_name))

    def __str__(self):
        return str(self.record_2_dict())

    def merge_record(self, record):
        """
        merge with another record
        :param record:
        :return:
        """
        for attr_name in record.__dict__:
            setattr(self, attr_name, getattr(record, attr_name))
        return self

    def set_attr(self, attr_name, attr_value):
        """
        :param attr_name:
        :param attr_value:
        :return:
        """
        if attr_name in self.__dict__:
            setattr(self, attr_name, attr_value)
        else:
            raise RuntimeError("can't set a attribute which is not exist")

    def __str__(self):
        return "ExperimentRecord[{}]".format(",".join(self.attr_names))

    @classmethod
    def get_class_name(cls):
        return "ExperimentRecord[{}]".format(",".join(self.attr_names))
