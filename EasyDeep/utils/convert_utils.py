import numpy as np
import torch
from PIL import Image
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


def df_2_np(df):
    return df.values


def np_2_df(np):
    import pandas as pd
    return pd.DataFrame(np)
