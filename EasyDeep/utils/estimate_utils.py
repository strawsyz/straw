import os

import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt


def bbox_iou(box1, box2):
    """
    获得两个box的iou值
    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1,
                                                                                     min=0)

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    # 计算iou
    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou


def calcu_iou(pred, target, target_value=255):
    """
    基于每个像素计算iou
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


def iou_mean(pred, target, n_classes=1):
    # for mask and ground-truth label, not probability map
    ious = []
    ious_sum = 0
    pred = pred.view(-1)
    target = target.view(-1)

    for cls in range(1, n_classes + 1):
        pred_inds = pred == cls
        target_inds = target == cls

        intersection = (pred_inds[target_inds]).long().sum().data.item()  # Cast to long to prevent overflows
        union = pred_inds.long().sum().data.item() + target_inds.long().sum().data.item() - intersection
        if union == 0:
            # 分母会变成0
            ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / float(max(union, 1)))
            ious_sum += float(intersection) / float(max(union, 1))
    return ious_sum / n_classes


def calcu_iou_with_threshold(pred_part, gt_part, threshold=127, draw=False):
    pred_part = pred_part.copy()
    pred_part[pred_part >= threshold] = 255
    pred_part[pred_part < threshold] = 0

    res = calcu_iou(pred_part, gt_part)
    if draw:
        ax = plt.subplot("121")
        ax.imshow(gt_part, cmap="gray")
        ax = plt.subplot("122")
        ax.imshow(pred_part, cmap="gray")
        plt.show()

    return res


def analysis_iou_by_threshold(gt_image_paths, predict_image_paths, thresholds=range(1, 255), draw_every_iou=False,
                              is_draw_image=False):
    assert (len(gt_image_paths) == len(predict_image_paths))
    # plt.figure()
    all_ious = []
    for index, (gt_image_path, predict_image_path) in enumerate(zip(gt_image_paths, predict_image_paths)):
        gt_image = np.array(Image.open(gt_image_path).convert("L"))
        repdict_image = np.array(Image.open(predict_image_path).convert("L"))
        ious = [0 for _ in range(len(thresholds))]
        for index, threshold in enumerate(thresholds):
            cur_iou = calcu_iou_with_threshold(repdict_image, gt_image, threshold=threshold, draw=is_draw_image)
            ious[index] = cur_iou
        all_ious.append(ious)
        if draw_every_iou:
            plt.plot(thresholds, ious, label="image{}".format(index))
    iousum = np.array([0 for _ in range(len(ious))], dtype=np.float32)
    for ious in all_ious:
        iousum += ious
    print("best is threshold is {}".format(thresholds[np.argmax(iousum)]))
    average_iou = iousum / len(gt_image_paths)
    print("average calcu_iou is {}".format(average_iou))
    print("max average calcu_iou is {}".format(np.max(average_iou)))
    plt.plot(thresholds, ious, label="threshold-ious")

    plt.xlabel("threshold")
    plt.ylabel("iou")
    plt.title("threshold-iou")
    plt.legend()
    plt.show()
    return thresholds[np.argmax(iousum)], np.max(iousum) / len(gt_image_paths)


def iou_estimate(gt_dir, predict_dir, thresholds=range(1, 255), draw_every_iou=False,
                 is_draw_image=False):
    gt_paths = []
    predict_paths = []

    for file_name in os.listdir(predict_dir):
        gt_paths.append(os.path.join(gt_dir, file_name))
        predict_paths.append(os.path.join(predict_dir, file_name))
    analysis_iou_by_threshold(gt_paths, predict_paths, thresholds, draw_every_iou,
                              is_draw_image)


if __name__ == '__main__':
    gt_dir = "/home/straw/Downloads/dataset/polyp/TMP/07/mask"
    # predict_dir = "/home/straw/Download\models\polyp\\result/2020-08-09/"
    predict_dir = "/home/straw/Download\models\polyp\\result/2020-09-01/"
    iou_estimate(gt_dir, predict_dir)
