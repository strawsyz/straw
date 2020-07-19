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


def calcu_iou(pred, target, target_value=255):
    """基于每个像素计算iou"""
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


def my_iou(pred_part, gt_part, threshold=127, draw=False):
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


def analysis_iou_by_threshold(gt_image_paths, predict_image_paths, thresholds=range(1, 256)):
    assert (len(gt_image_paths) == len(predict_image_paths))
    plt.figure()
    all_ious = []
    for index, (gt_image_path, predict_image_path) in enumerate(zip(gt_image_paths, predict_image_paths)):
        gt_image = np.array(Image.open(gt_image_path).convert("L"))
        repdict_image = np.array(Image.open(predict_image_path).convert("L"))
        ious = [0 for _ in range(255)]
        for index, threshold in enumerate(thresholds):
            threshold += 1
            cur_iou = my_iou(repdict_image, gt_image, threshold=threshold)
            ious[index] = cur_iou
        all_ious.append(ious)
        plt.plot(thresholds, ious, label="image{}".format(index))
    iousum = np.array([0 for _ in range(len(ious))], dtype=np.float32)
    for ious in all_ious:
        iousum += ious
    print("best is threshold is {}".format(thresholds[np.argmax(iousum)]))
    print(iousum)
    print("average calcu_iou is {}".format(np.max(iousum) / len(gt_image_paths)))
    plt.xlabel("threshold")
    plt.ylabel("iou")
    plt.title("threshold-iou")
    plt.legend()
    plt.show()
    return thresholds[np.argmax(iousum)], np.max(iousum) / len(gt_image_paths)


def analysis_iou_by_threshold4unet(paths):
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
    print("average calcu_iou is {}".format(np.max(iousum) / len(paths)))
    plt.xlabel("threshold")
    plt.ylabel("calcu_iou")
    plt.title("threshold-calcu_iou")
    plt.legend()
    plt.show()


def get_image(result_path, ind, pattern):
    path = os.path.join(result_path, pattern.format(ind))
    return np.array(Image.open(path))


pattern_dict = {"GT": "{}_M.png",
                "o": "{}_testT.png",
                "pGT": "{}_mask.png",
                "0": "{}_test.png",
                "1": "{}_test2.png",
                "2": "{}_test3.png",
                "3": "{}_test4.png"}


def analysis_iou_on_miyazakidata(result_path, pattern="GT_o"):
    """对最终4张图片统合的结果进行测试"""
    ious = []
    pattern1, pattern2 = pattern.strip().split("_")
    pattern1 = pattern_dict[pattern1]
    pattern2 = pattern_dict[pattern2]
    for i in range(1, 21):
        image1 = get_image(result_path, i, pattern1)
        image2 = get_image(result_path, i, pattern2)
        # handled_gt_path = os.path.join(result_path, "{}_mask.png".format(i))
        # gt_path = os.path.join(result_path, "{}_M.png".format(i))
        # image_path = os.path.join(result_path, "{}_testT.png".format(i))
        # gt_path = np.array(Image.open(gt_path))
        # image_path = np.array(Image.open(image_path))
        ious.append(calcu_iou(image1, image2))
    print(ious)
    print("average of iou is {:.4f}".format(np.mean(ious)))

    return ious


def iou_estimate(gt_dir, predict_dir):
    gt_paths = []
    predict_paths = []

    for file_name in os.listdir(predict_dir):
        gt_paths.append(os.path.join(gt_dir, file_name))
        predict_paths.append(os.path.join(predict_dir, file_name))
    analysis_iou_by_threshold(gt_paths, predict_paths)


import os

if __name__ == '__main__':
    # paths = [os.path.join("miyazaki", file) for file in os.listdir("miyazaki")]
    # analysis_iou_by_threshold4unet(paths)

    gt_dir = "D:\Download\datasets\polyp\\06\mask"
    predict_dir = "D:\Download\models\polyp\\result\\2020-07-19"
    iou_estimate(gt_dir, predict_dir)

    # image_path = "D:\Download\datasets\polyp\\06\mask\Proc201506020034_1_2_1.png"
    # predict_path = "D:\Download\models\polyp\\result\\2020-07-19\Proc201506020034_1_2_1.png"
    # analysis_iou_by_threshold([image_path], [predict_path], thresholds=[0])

    # result_path = "D:\(lab\graduate\プログラム\精度評価\\result8"
    #
    # fig = plt.figure()  # 创建画布
    # ax = plt.subplot()  # 创建作图区域
    # patterns = ["GT_o", "pGT_o", "GT_0", "GT_1", "GT_2", "GT_3"]
    # ious4pattern = []
    # for pattern in patterns:
    #     ious = analysis_iou_on_miyazakidata(result_path, pattern)
    #     ious4pattern.append(ious)
    # ax.boxplot(ious4pattern, whis=[5, 95])  # 设置最大值不超过95分位点；最小值不小于5%分位点。
    # ax.set_xticklabels(patterns)
    # plt.show()
# best is threshold is 99
# average calcu_iou is 0.40121014292501733
