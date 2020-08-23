import numpy as np
from PIL import Image
import torch
from torch.autograd import Variable


def image_2_np(image):
    return np.array(image)


def np_2_image(narray):
    return Image.fromarray(narray)


def tensor_2_np(tensor):
    return tensor.numpy()


def np_2_tensor(narray):
    return torch.from_numpy(narray)


def np_2_variable(narray):
    return Variable(torch.from_numpy(narray))


def varibale_2_np(varibale):
    return varibale.data.numpy()
