# 将一些转换函数保存起来
#
import numpy as np
from PIL import Image


def Image2np(image):
    return np.array(image)


def np2Image(narray):
    return Image.fromarray(narray)
