import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def bbox_iou(box1, box2):
    import torch
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
    # Intersection area 这里没有对inter_area为负的情况进行判断，后面计算出来的IOU就可能是负的
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1,
                                                                                     min=0)

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    # 计算iou
    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou


def iou(pred, target):
    pred_mask = pred == 255
    # print("image size is {}".format(pred.size))
    # print((pred == 255).sum())
    # print((target == 255).sum())
    # target[]
    inter_mask = target[pred_mask] == 255
    inter_size = (inter_mask.sum())
    out_size = (pred == 255).sum() + (target == 255).sum() - inter_size
    # print("inner size is : {}".format(inter_size))
    # print("out size is : {}".format(out_size))
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


def my_iou(pred_part, gt_part, threshold=127, draw=False):
    pred_part = pred_part.copy()
    pred_part[pred_part >= threshold] = 255
    pred_part[pred_part < threshold] = 0

    res = iou(pred_part, gt_part)
    # print("iou is {}".format(res))
    if draw:
        ax = plt.subplot("121")
        ax.imshow(gt_part, cmap="gray")
        ax = plt.subplot("122")
        ax.imshow(pred_part, cmap="gray")
        plt.show()

    return res


def analysis_iou_by_threshold(paths):
    plt.figure()
    all_ious = []
    for index, path in enumerate(paths):
        image = Image.open(path)

        image = np.array(image)
        heights = range(0, 1752, 584)
        height_start = heights[1]
        gt_part = image[height_start:height_start + 584, 0:565]
        height_start = heights[2]
        pred_part = image[height_start:height_start + 584, 0:565]
        ious = [0 for _ in range(256)]
        for threshold in range(256):
            cur_iou = my_iou(pred_part, gt_part, threshold=threshold)
            ious[threshold] = cur_iou
        all_ious.append(ious)
        plt.plot(range(256), ious, label="image{}".format(index))
    iousum = np.array([0 for _ in range(len(ious))], dtype=np.float64)
    for ious in all_ious:
        iousum += ious
    print("best is threshold is {}".format(np.argmax(iousum)))
    print("average iou is {}".format(np.max(iousum) / len(paths)))
    plt.xlabel("threshold")
    plt.ylabel("iou")
    plt.title("threshold-iou")
    plt.legend()
    plt.show()


import os

if __name__ == '__main__':
    paths = [os.path.join("miyazaki", file) for file in os.listdir("miyazaki")]
    # paths = ["temp{}.png".format(x) for x in range(5)]
    analysis_iou_by_threshold(paths)
# best is threshold is 99
# average iou is 0.40121014292501733
