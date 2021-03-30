import numpy as np
from PIL import Image
import torch
from torch.autograd import Variable


def image_2_np(image):
    return np.array(image)


def np_2_image(np_arr):
    return Image.fromarray(np_arr)


def tensor_2_np(tensor):
    return tensor.numpy()


def np_2_tensor(np_arr):
    return torch.from_numpy(np_arr)


def np_2_variable(np_arr):
    return Variable(torch.from_numpy(np_arr))


def variable_2_np(variable):
    return variable.data.numpy()
