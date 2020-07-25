import torch

from base.base_checkpoint import BaseCheckPoint
from utils.utils_ import copy_attr


def save(checkpoint, path):
    torch.save(checkpoint, path)


def load(checkpoint, path):
    copy_attr(torch.load(path), checkpoint)


if __name__ == '__main__':
    model_path = "D:\Download\models\polyp\ep825_07-17-43.pkl"
    checkpoint = BaseCheckPoint()
    load(checkpoint, model_path)
    current_epoch = checkpoint.epoch + 1
    save(checkpoint.state_dict, "temp_state.pkl")
    save(checkpoint.optimizer, "optimizer.pkl")

    # net.load_state_dict(checkpoint.state_dict)
    # self.optimizer.load_state_dict(checkpoint.optimizer)
