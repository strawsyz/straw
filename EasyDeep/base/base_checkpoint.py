from builtins import dict

import torch


class InvalidAttr(Exception):
    def __init__(self, attr_name):
        message = "attr:{} is not exist!".format(attr_name)
        super(InvalidAttr, self).__init__(message)


class BaseCheckPoint:

    def __init__(self):
        super(BaseCheckPoint, self).__init__()
        self.state_dict = None
        self.epoch = None
        # 如果优化器是SGD的话，就不需要保存
        self.optimizer = None

    def __call__(self, data: dict):
        for attr in data:
            if hasattr(self, attr):
                setattr(self, attr, data[attr])
            else:
                raise InvalidAttr(attr)
        return self

    @staticmethod
    def create_checkpoint(data: dict):
        return BaseCheckPoint()(data)


class CustomCheckPoint(BaseCheckPoint):
    def __init__(self, attr_names: list = []):
        for attr_name in attr_names:
            setattr(self, attr_name, None)


def save(checkpoint, path):
    torch.save(checkpoint, path)


from utils.utils_ import copy_attr


def load(checkpoint, path):
    copy_attr(torch.load(path), checkpoint)
