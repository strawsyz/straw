import cv2
import random
import torchvision.transforms.functional as functional
from torchvision import transforms
import numpy as np
from PIL import Image


def rotate4_imglist(images):
    """对多张图片旋转一个相同的随机的角度"""
    # 拿到角度的随机数。angle是一个-180到180之间的一个数
    angle = transforms.RandomRotation.get_params([-180, 180])
    # 对image和mask做相同的旋转操作，保证他们都旋转angle角度
    for ind, image in enumerate(images):
        images[ind] = image.rotate(angle)
    # mask = mask.rotate(angle)
    return images


def vflip_imagelist(images):
    # 50%的概率应用垂直，水平翻转。
    if random.random() < 0.5:
        for i, image in enumerate(images):
            images[i] = functional.vflip(image)
    return images


def hflip_imagelist(images):
    # 50%的概率应用垂直，水平翻转。
    if random.random() < 0.5:
        for i, image in enumerate(images):
            images[i] = functional.hflip(image)
    return images


def detect_edge_by_canny(source_path, save_path):
    img = cv2.imread(source_path)
    img = cv2.resize(img, (224, 224))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # canny_gray(gray)
    # detected_edges = cv2.GaussianBlur(gray, (3, 3), 0)
    detected_edges = cv2.Canny(gray, 60, 110, apertureSize=3)
    # just add some colours to edges from original image.
    dst = cv2.bitwise_and(img, img, mask=detected_edges)
    # cv2.imshow('canny demo', dst)
    # print(type(dst))
    cv2.imwrite(save_path, dst)


def find_contours(binary_image_path, margin=12, contours_color=127):
    image = cv2.imread(binary_image_path, cv2.IMREAD_GRAYSCALE)
    copy_image = image.copy()

    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(image, contours, -1, (contours_color), margin)
    half_margin = int(margin / 2) + 1
    copy_image[half_margin:-half_margin, half_margin:-half_margin] = image[half_margin:-half_margin,
                                                                     half_margin:-half_margin]
    return copy_image


def _add_margin(image, margin):
    width, height = image.size
    new_image = Image.new('RGB', (width + margin * 2, height + margin * 2), color=(0, 0, 0))
    new_image.paste(image, (margin, margin))
    return np.array(new_image)


def _remove_margin(image_array, margin):
    return image_array[margin:-margin, margin:-margin, :]


