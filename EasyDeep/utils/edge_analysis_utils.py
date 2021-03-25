#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2021/03/14 21:30
# @Author  : strawsyz
# @File    : edge_analysis_utils.py
# @desc:

import cv2


def calcu_iou(pred, target, target_value=255):
    """
    calculate IoU by pixel
    :param pred: predict result
    :param target: ground truth
    :param target_value:
    :return:
    """
    pred_mask = pred == target_value
    inter_mask = target[pred_mask] == target_value
    inter_size = (inter_mask.sum())
    out_size = (pred == target_value).sum() + (target == target_value).sum() - inter_size
    return inter_size / out_size


# 创建图像的轮廓图像
def edge_analysis(source_path, mask_path):
    img = cv2.imread(source_path)
    mask_image = cv2.imread(mask_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # canny_gray(gray)
    # detected_edges = cv2.GaussianBlur(gray, (3, 3), 0)
    # detected_edges = cv2.Canny(gray, 60, 110, apertureSize=3)
    detected_edges = cv2.Canny(gray, 0, 0)    # .3580877798795899

    # just add some colours to edges from original image.
    dst = cv2.bitwise_and(img, img, mask=detected_edges)
    dst_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    # dst_gray[dst_gray < threshold] = 0
    dst_gray[dst_gray >= threshold] = 255

    IoU = calcu_iou(dst_gray, mask_image)
    # print(IoU)
    return IoU
    # cv2.imshow('canny demo', dst)
    # print(type(dst))
    # cv2.imwrite(save_path, dst_gray)


if __name__ == '__main__':
    threshold = 1
    root_path = r"/home/shi/Downloads/dataset/polyp/TMP/09/train/data"
    mask_root_path = r"/home/shi/Downloads/dataset/polyp/TMP/09/train/mask"
    import os

    all_IoU = 0
    for filename in os.listdir(root_path):
        source_path = os.path.join(root_path, filename)
        mask_path = os.path.join(mask_root_path, filename)
        IoU = edge_analysis(source_path, mask_path)
        all_IoU += IoU
    print(all_IoU / len(os.listdir(root_path)))
