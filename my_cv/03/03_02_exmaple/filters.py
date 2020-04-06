import cv2
import numpy as np
import utils


def stroke_edges(src, dst, blur_ksize=7, edge_ksize=5):
    if blur_ksize >= 3:
        # 提示：对于较大的ksize，使用medianBlur的代价很高
        # medianBlur作为模糊函数，用于去噪
        blurred_src = cv2.medianBlur(src, blur_ksize)
        gray_src = cv2.cvtColor(blurred_src, cv2.COLOR_BGR2GRAY)
    else:
        gray_src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    # 使用Laplacian作为边缘检测函数
    # cv2.CV_8U表示目标图像每个通道为24位
    # 如果为-1表示目标图像和换图像有同样的位深度
    cv2.Laplacian(gray_src, cv2.CV_8U, gray_src, ksize=edge_ksize)
    # 归一化
    normalizedInverseAlpha = (1.0 / 255) * (255 - gray_src)
    channels = cv2.split(src)
    for channels in channels:
        # 乘以原图像，将边缘变黑
        channels[:] = channels * normalizedInverseAlpha
    cv2.merge(channels, dst)


class VConvolutionFilter(object):
    def __init__(self, kernel):
        self._kernel = kernel

    def apply(self, src, dst):
        cv2.filter2D(src, -1, self._kernel, dst)


# 注意！ 权重加起来为一，这样的话不会改变图像亮度
class SharpenFilter(VConvolutionFilter):
    """锐化滤波器"""

    def __init__(self):
        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])
        VConvolutionFilter.__init__(self, kernel)


# 改一下锐化核，是得权重和为0，就能得到边缘检测核
# 将边缘变白色，非边缘变黑
class FindEdgesFilter(VConvolutionFilter):
    """边缘检测滤波器"""

    def __init__(self):
        kernel = np.array([[-1, -1, -1],
                           [-1, 8, -1],
                           [-1, -1, -1]])
        VConvolutionFilter.__init__(self, kernel)


class BlurFilter(VConvolutionFilter):
    """邻近平均滤波器"""

    # 通常权重和为1，且邻接像素的权重权威正
    def __init__(self):
        kernel = np.array([[0.04, 0.04, 0.04, 0.04, 0.04],
                           [0.04, 0.04, 0.04, 0.04, 0.04],
                           [0.04, 0.04, 0.04, 0.04, 0.04],
                           [0.04, 0.04, 0.04, 0.04, 0.04],
                           [0.04, 0.04, 0.04, 0.04, 0.04]])


class EmbossFilter(VConvolutionFilter):
    """浮雕滤波器，能产生浮雕的效果"""

    def __init__(self):
        kernel = np.array([-2, -1, 0],
                          [-1, 1, 1],
                          [0, 1, 2])
        VConvolutionFilter.__init__(self, kernel)


class BGRFuncFilter():
    def __init__(self, v_func=None, b_func=None, g_func=None, r_func=None, dtype=np.uint8):
        length = np.iinfo(dtype).max + 1
        self._b_lookup_array = utils.create_lookup_array(utils.create_composite_func(b_func, v_func), length)
        self._g_lookup_array = utils.create_lookup_array(utils.create_composite_func(g_func, v_func), length)
        self._r_lookup_array = utils.create_lookup_array(utils.create_composite_func(r_func, v_func), length)

    def apply(self, src, dst):
        """将滤波器应用到BGR的src和dst中"""
        b, g, r = cv2.split(src)
        utils.apply_lookup_array(self._b_lookup_array, b, b)
        utils.apply_lookup_array(self._g_lookup_array, g, g)
        utils.apply_lookup_array(self._r_lookup_array, r, r)
        cv2.merge([b, g, r], dst)


class BGRCurveFilter(BGRFuncFilter):
    def __init__(self, v_points=None, b_points=None, g_points=None, r_points=None, dtype=np.uint8):
        BGRFuncFilter.__init__(self, utils.create_curve_func(v_points), utils.create_curve_func(b_points),
                               utils.create_curve_func(g_points), utils.create_curve_func(r_points), dtype)


class BGRPortraCurveFilter(BGRCurveFilter):
    def __init__(self, dtype=np.uint8):
        BGRCurveFilter.__init__(self, v_points=[(0, 0), (23, 20), (157, 173), (255, 255)],
                                b_points=[(0, 0), (41, 46), (231, 228), (255, 255)],
                                g_points=[(0, 0), (52, 47), (189, 196), (255, 255)],
                                r_points=[(0, 0), (69, 69), (213, 218), (255, 255)], dtype=dtype)
