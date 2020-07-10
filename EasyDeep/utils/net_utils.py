import torch

from base.base_checkpoint import BaseCheckPoint
from utils.utils_ import copy_attr


def save(checkpoint, path):
    torch.save(checkpoint, path)


def load(checkpoint, path) -> BaseCheckPoint:
    copy_attr(torch.load(path), checkpoint)
