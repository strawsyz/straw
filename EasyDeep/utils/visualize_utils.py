#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/2/3 15:10
# @Author  : strawsyz
# @File    : visualize.py
# @desc:
from matplotlib import pyplot as plt


def show_images(images, labels=None):
    if labels is None:
        labels = range(len(images))
    import math
    row_num = int(math.sqrt(len(images)))
    col_num = int(len(images) / row_num) + 1 if len(images) % row_num != 0 else int(len(images) / row_num)
    fig, axs = plt.subplots(row_num, col_num)
    if len(images) > 1:
        import numpy as np
        axs = np.asarray(axs)
        axs = axs.flatten()
        for idx in range(len(images)):
            axs[idx].imshow(images[idx].squeeze(), cmap='gray')
            axs[idx].set_title(labels[idx])
            axs[idx].axis("off")
    else:
        axs.imshow(images[0].squeeze(), cmap='gray')
        axs.set_title(labels[0])
        axs.axis("off")
    plt.show()
