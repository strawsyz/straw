import torch
from PIL import Image


def get_image(path="sample.jpg", mode="RGB"):
    """ get a image sample"""
    if mode is None:
        return Image.open(path)
    else:
        return Image.open(path).convert(mode)


def get_tensor(*args):
    return torch.randn(*args)