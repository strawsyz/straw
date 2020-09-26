import cv2
import random
import torchvision.transforms.functional as functional
from torchvision import transforms
import numpy as np
from PIL import Image


def rotate_images(images):
    angle = transforms.RandomRotation.get_params([-180, 180])
    for ind, image in enumerate(images):
        images[ind] = image.rotate(angle)
    return images


def rotate_images(images, probability=0.5):
    if random.random() < probability:
        for i, image in enumerate(images):
            images[i] = functional.vflip(image)
    return images


def rotate_images(images, probability=0.5):
    if random.random() < probability:
        for i, image in enumerate(images):
            images[i] = functional.hflip(image)
    return images


def detect_edge_by_canny(source_path, save_path):
    img = cv2.imread(source_path)
    img = cv2.resize(img, (224, 224))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detected_edges = cv2.Canny(gray, 60, 110, apertureSize=3)
    # add some colours to edges from original image.
    dst = cv2.bitwise_and(img, img, mask=detected_edges)
    # cv2.imshow('canny demo', dst)
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


def binary_image(image_array, threshold=127, max_value=255):
    ret, thresh1 = cv2.threshold(image_array, threshold, max_value, cv2.THRESH_BINARY)
    return thresh1
