import cv2
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

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



def compare_images(rootpath1, rootpath2):
    import os
    for filename in os.listdir(rootpath1):
        filepath = os.path.join(rootpath1, filename)
        filepath2 = os.path.join(rootpath2, filename)
        image = cv2.imread(filepath)
        image2 = cv2.imread(filepath2)
        if image.shape != image2.shape:
            raise RuntimeWarning("Size is different")


def pd_profiling(data_frame, output_file="pandas_profiling.html"):
    import pandas_profiling
    profile = pandas_profiling.ProfileReport(data_frame)
    profile.to_file(output_file=output_file)


def analyze_num_list(data: list):
    print(f"min : {min(data)}")
    print(f"max : {max(data)}")
    print(f"mean : {np.mean(data)}")
    print(f"median : {np.median(data)}")
    from scipy import stats
    print(f"mode : {stats.mode(data)[0][0]}")
