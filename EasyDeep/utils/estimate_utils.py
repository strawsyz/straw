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
    # best is threshold is 1
    # [2.7021086e+01 2.6735561e+01 2.6445284e+01 2.6153549e+01 2.5852694e+01
    #  2.5551973e+01 2.5247835e+01 2.4939827e+01 2.4632431e+01 2.4316172e+01
    #  2.3990261e+01 2.3668369e+01 2.3346140e+01 2.3021303e+01 2.2688931e+01
    #  2.2355656e+01 2.2019621e+01 2.1687687e+01 2.1350330e+01 2.1004524e+01
    #  2.0660049e+01 2.0319241e+01 1.9970459e+01 1.9621588e+01 1.9270105e+01
    #  1.8921432e+01 1.8569300e+01 1.8212616e+01 1.7861862e+01 1.7506281e+01
    #  1.7146090e+01 1.6792110e+01 1.6438992e+01 1.6079981e+01 1.5724097e+01
    #  1.5372369e+01 1.5017054e+01 1.4673626e+01 1.4327996e+01 1.3985425e+01
    #  1.3644767e+01 1.3309999e+01 1.2970323e+01 1.2640488e+01 1.2311566e+01
    #  1.1988156e+01 1.1672306e+01 1.1359271e+01 1.1057635e+01 1.0761332e+01
    #  1.0464785e+01 1.0172806e+01 9.8879385e+00 9.6099100e+00 9.3380413e+00
    #  9.0672159e+00 8.7986393e+00 8.5391827e+00 8.2900391e+00 8.0458069e+00
    #  7.8048487e+00 7.5656109e+00 7.3346415e+00 7.1052113e+00 6.8884768e+00
    #  6.6791358e+00 6.4727454e+00 6.2744446e+00 6.0758896e+00 5.8865585e+00
    #  5.6974487e+00 5.5151944e+00 5.3404078e+00 5.1715236e+00 5.0084038e+00
    #  4.8508897e+00 4.6973243e+00 4.5485635e+00 4.4085422e+00 4.2684216e+00
    #  4.1341777e+00 4.0012207e+00 3.8740962e+00 3.7536316e+00 3.6352310e+00
    #  3.5213938e+00 3.4115098e+00 3.3039043e+00 3.2024717e+00 3.1021543e+00
    #  3.0072913e+00 2.9123397e+00 2.8201935e+00 2.7325070e+00 2.6485188e+00
    #  2.5630984e+00 2.4810836e+00 2.4010792e+00 2.3271620e+00 2.2530043e+00
    #  2.1792812e+00 2.1095788e+00 2.0396221e+00 1.9719558e+00 1.9062493e+00
    #  1.8432965e+00 1.7805163e+00 1.7212377e+00 1.6606519e+00 1.6006006e+00
    #  1.5453392e+00 1.4878149e+00 1.4341158e+00 1.3803372e+00 1.3276888e+00
    #  1.2788657e+00 1.2297957e+00 1.1817527e+00 1.1369541e+00 1.0941544e+00
    #  1.0502409e+00 1.0086488e+00 9.7118473e-01 9.3230569e-01 8.9516276e-01
    #  8.6079443e-01 8.2586575e-01 7.9135549e-01 7.5904787e-01 7.2791088e-01
    #  6.9719952e-01 6.6765857e-01 6.3996452e-01 6.1308926e-01 5.8722430e-01
    #  5.6257552e-01 5.3762424e-01 5.1217419e-01 4.8904267e-01 4.6570250e-01
    #  4.4440815e-01 4.2241025e-01 4.0143538e-01 3.8232577e-01 3.6386627e-01
    #  3.4560281e-01 3.2886446e-01 3.1206232e-01 2.9684591e-01 2.8188479e-01
    #  2.6867068e-01 2.5550774e-01 2.4297354e-01 2.3022044e-01 2.1827514e-01
    #  2.0790727e-01 1.9823122e-01 1.8903755e-01 1.8060634e-01 1.7220284e-01
    #  1.6390480e-01 1.5661222e-01 1.4988467e-01 1.4302762e-01 1.3646257e-01
    #  1.2977158e-01 1.2428287e-01 1.1922436e-01 1.1354357e-01 1.0878797e-01
    #  1.0321430e-01 9.8576695e-02 9.3877986e-02 9.0167679e-02 8.6612463e-02
    #  8.3397754e-02 8.0268003e-02 7.6866023e-02 7.4366570e-02 7.1453631e-02
    #  6.9640316e-02 6.7210175e-02 6.4973876e-02 6.3142993e-02 6.0866635e-02
    #  5.8611974e-02 5.6527596e-02 5.4851335e-02 5.2737687e-02 5.1211212e-02
    #  4.9273409e-02 4.7782492e-02 4.5883741e-02 4.4354703e-02 4.2910464e-02
    #  4.1313492e-02 3.9961312e-02 3.8761951e-02 3.7637357e-02 3.6332712e-02
    #  3.5073418e-02 3.3870339e-02 3.2569610e-02 3.1343561e-02 3.0358626e-02
    #  2.9369729e-02 2.8332014e-02 2.7419506e-02 2.6513807e-02 2.5443289e-02
    #  2.4416020e-02 2.3536693e-02 2.2710217e-02 2.1866683e-02 2.1013835e-02
    #  2.0072328e-02 1.9251961e-02 1.8467149e-02 1.7670602e-02 1.6843449e-02
    #  1.6169190e-02 1.5378813e-02 1.4702525e-02 1.3958970e-02 1.3228724e-02
    #  1.2546341e-02 1.1886201e-02 1.1459109e-02 1.0927424e-02 1.0343143e-02
    #  1.0021985e-02 9.5115406e-03 8.9199943e-03 8.4281918e-03 8.0203190e-03
    #  7.4085505e-03 6.7470605e-03 6.2662396e-03 5.7407110e-03 5.4134391e-03
    #  5.0665238e-03 4.4819070e-03 4.0163756e-03 3.6846667e-03 3.0400176e-03
    #  2.6232395e-03 2.2175023e-03 1.8542963e-03 1.6618745e-03 1.4549762e-03
    #  1.0568356e-03 7.2671578e-04 4.7446167e-04 3.0716445e-04 0.0000000e+00]
    # average calcu_iou is 0.31789512634277345

    #     image_path = "D:\Download\datasets\polyp\\06\mask\Proc201506020034_1_2_1.png"
    #     predict_path = "D:\Download\models\polyp\\result\\2020-07-19\Proc201506020034_1_2_1.png"
    #     analysis_iou_by_threshold([image_path], [predict_path], thresholds=[0])


def estimate_1():
    result_path = "D:\(lab\graduate\プログラム\精度評価\\result8"

    fig = plt.figure()  # 创建画布
    ax = plt.subplot()  # 创建作图区域
    patterns = ["GT_o", "pGT_o", "GT_0", "GT_1", "GT_2", "GT_3"]
    ious4pattern = []
    for pattern in patterns:
        ious = analysis_iou_on_miyazakidata(result_path, pattern)
        ious4pattern.append(ious)
    ax.boxplot(ious4pattern, whis=[5, 95])  # 设置最大值不超过95分位点；最小值不小于5%分位点。
    ax.set_xticklabels(patterns)
    plt.show()
    # best is threshold is 99
    # average calcu_iou is 0.40121014292501733
