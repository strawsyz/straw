import torch

from utils.common_utils import copy_attr


def save(checkpoint, path):
    torch.save(checkpoint, path)


def load(checkpoint, path):
    copy_attr(torch.load(path), checkpoint)
