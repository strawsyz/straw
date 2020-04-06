import cv2
import numpy as np
import utils


def copy_rect(src, dst, scr_rect, dst_rect, mask=None, interpolation=cv2.INTER_LINEAR):
    """复制源矩形到目标矩形"""
    x_src, y_src, w_src, h_src = scr_rect
    x_dst, y_dst, w_dst, h_dst = dst_rect

    if mask is None:
        # 将源矩形大小变为目标矩形大小
        dst[y_dst:y_dst + h_dst, x_dst:x_dst + w_dst] = \
            cv2.resize(src[y_src:y_src + h_src, x_src:x_src + w_src],
                       (w_dst, h_dst),
                       interpolation=interpolation)
    else:
        if not utils.is_gray(src):
            # 将mask变为3个通道
            mask = mask.repeat(3).reshape(h_src, w_src, 3)
        # 根据mask确定哪些像素需要复制
        dst[y_dst:y_dst + h_dst, x_dst:x_dst + w_dst] = \
            np.where(cv2.resize(mask, (w_dst, h_dst),
                                interpolation=cv2.INTER_NEAREST),
                     cv2.resize(
                         src[y_src:y_src + h_src, x_src:x_src + w_src],
                         (w_dst, h_dst),
                         interpolation=interpolation),
                     dst[y_dst:y_dst + h_dst, x_dst:x_dst + w_dst])


def swap_rects(src, dst, rects, masks=None, interpolation=cv2.INTER_LINEAR):
    if dst is not src:
        dst[:] = src

    num_rects = len(rects)

    if num_rects < 2:
        return

    if masks is None:
        masks = [None] * num_rects

    # 将原图片的最后一个矩形中的内容临时存储起来
    x, y, w, h = rects[num_rects - 1]
    temp = src[y:y + h, x:x + w].copy()

    i = num_rects - 2
    # 复制每一个矩形到下一个矩形
    while i >= 0:
        copy_rect(src, dst, rects[i], rects[i + 1], masks[i])
        i -= 1
    # 复制存储起来的最后一个矩形的内容到目标图片的第一个矩形
    copy_rect(temp, dst, (0, 0, w, h), rects[0], masks[num_rects - 1], interpolation)
