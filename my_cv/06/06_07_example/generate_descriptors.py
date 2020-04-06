import os
import cv2
import numpy as np
from os import walk
from os.path import join
import sys

"""将图片的描述符保存到文件中
使用方法： python generate_descriptors.py floder_path"""


def create_descriptors(folder):
    files = []
    for (dirpath, dirnames, filenames) in walk(folder):
        files.extend(filenames)
    for f in files:
        # 防止win系统中的缩略图也被处理
        if f != "Thumbs.db":
            save_descriptor(folder, f, cv2.xfeatures2d.SIFT_create())


def save_descriptor(folder, image_path, feature_detector):
    print("reading %s" % image_path)
    if image_path.endswith("npy"):
        return
    img = cv2.imread(join(folder, image_path), 0)
    keypoints, descriptors = feature_detector.detectAndCompute(img, None)
    descriptor_file_path = image_path.replace("jpg", "npy")
    # numpy中的save方法可以采用优化方式将数据数据保存到一个文件中
    # 这里是将扫描符保存到文件中
    np.save(join(folder, descriptor_file_path), descriptors)


dir_name = sys.argv[1]

create_descriptors(dir_name)
