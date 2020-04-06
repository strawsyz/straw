import cv2
import numpy as np


def created_median_mask(disparity_map, valid_depth_mask, rect=None):
    """生成掩模，使得矩形中不想要的区域的掩模值为0，想要的区域的掩模值为1"""
    if rect is not None:
        x, y, w, h = rect
        disparity_map = disparity_map[y:y + h, x:x + w]
        valid_depth_mask = valid_depth_mask[y:y + h, x:x + w]
    # 获得中位数
    median = np.median(disparity_map)
    # 当有效的视差值与平均视差值相差12 或者更多时，可以将像素看做噪声。12 这个值是根据经验
    return np.where((valid_depth_mask == 0) | (abs(disparity_map - median) < 12), 1.0, 0.0)

