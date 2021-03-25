from matplotlib import pyplot as plt
import cv2
from PIL import Image


def img_hist_plt(gray_image_array):
    plt.figure()
    plt.hist(gray_image_array.flatten(), 128)
    plt.show()


def img_hist_cv2(gray_image_array):
    channels = cv2.split(gray_image_array)
    new_channels = []
    for channel in channels:
        new_channels.append(cv2.equalizeHist(channel))

    result = cv2.merge(tuple(new_channels))
    cv2.imshow("equalizeHist", result)

    cv2.waitKey(0)


def is_same_size(image_paths):
    """check if images have the same size"""
    size = Image.open(image_paths[0]).size
    print("normal size is {}".format(size))
    for image_path in image_paths[1:]:
        image = Image.open(image_path)
        if size != image.size:
            print("{} have different shape: {}".format(image_path, image.size))


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


def compare_images(rootpath1, rootpath2):
    import os
    for filename in os.listdir(rootpath1):
        filepath = os.path.join(rootpath1, filename)
        filepath2 = os.path.join(rootpath2, filename)
        image = cv2.imread(filepath)
        image2 = cv2.imread(filepath2)
        if image.shape != image2.shape:
            raise RuntimeWarning("Size is different")
