class EpochRecord:
    """记录一个epoch的训练结果"""

    def __init__(self, train_loss, valid_loss=None, config_desc=None):
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


class HistoryRecord:
    """记录一次实验的记录"""

    def __init__(self, train_loss, valid_loss=None, config_desc=None):
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.scores = ["train_loss", "valid_loss"]
        self.config_desc = config_desc

    def get_record_dict(self):
        record_dict = {}
        for attr in self.scores:
            record_dict[attr] = getattr(self, attr, None)
        return record_dict


class CustomEpochRecord:
    def __init__(self, attr_names: list = []):
        for attr_name in attr_names:
            setattr(self, attr_name, None)

    def add(self, record):
        # 如果新设置的record有重复的部分，会覆盖到之前的数据上
        for attr_name in record.__dict__:
            setattr(self, attr_name, getattr(record, attr_name))
        return self

    def set(self, attr_name, attr_value):
        if attr_name in self.__dict__:
            setattr(self, attr_name, attr_value)
        else:
            raise RuntimeError("can't set a attribute which is not exist")

    def get_record_dict(self):
        record_dict = {}
        for attr in self.__dict__:
            record_dict[attr] = getattr(self, attr, None)
        return record_dict

    def __call__(self, *args, **kwargs):
        for attr_name in kwargs:
            setattr(self, attr_name, kwargs.get(attr_name))

    def __str__(self):
        return str(self.get_record_dict())


if __name__ == '__main__':
    record = CustomEpochRecord(["train_loss", "valid_loss"])
    record.set("train_loss", 1)
    record.set("valid_loss", 2)
    record(start=2)
    print(record)
    record2 = CustomEpochRecord(["train_loss1", "train_loss2"])
    record2.set("train_loss2", 4)
    record.add(record2)
    record.set("train_loss", 3)
    print(record)
