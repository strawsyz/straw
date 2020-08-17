import torch

from utils.utils_ import copy_attr


def save(checkpoint, path):
    torch.save(checkpoint, path)


def load(checkpoint, path):
    copy_attr(torch.load(path), checkpoint)
