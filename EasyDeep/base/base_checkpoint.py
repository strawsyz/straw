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


if __name__ == '__main__':
    data = {"asd": 123, "epoch": 123}
    checkpoint = CustomCheckPoint(data.keys())(data)
    # 初始化
    # checkpoint = BaseCheckPoint.create_checkpoint(data)
    print(checkpoint.asd)
    # 保存
    save(checkpoint, "1.pth")
    # 读取
    load(checkpoint, "1.pth")
    print(checkpoint.asd)
