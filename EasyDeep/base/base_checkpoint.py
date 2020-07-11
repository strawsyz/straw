from builtins import dict

import torch


class UnValidAttr(Exception):
    def __init__(self):
        super(UnValidAttr, self).__init__()


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
                raise UnValidAttr
        return self

    @staticmethod
    def create_checkpoint(data: dict):
        return BaseCheckPoint()(data)


def save(checkpoint, path):
    torch.save(checkpoint, path)


from utils.utils_ import copy_attr


def load(checkpoint, path):
    copy_attr(torch.load(path), checkpoint)


if __name__ == '__main__':
    data = {"state_dict": 123, "epoch": 123}
    checkpoint = BaseCheckPoint.create_checkpoint(data)
    print(checkpoint.state_dict)
    save(checkpoint, "1.pth")
    load(BaseCheckPoint(), "1.pth")
    print(checkpoint.state_dict)
