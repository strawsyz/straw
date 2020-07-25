import cv2
import numpy as np


def tif2jpg(file_path):
    img = cv2.imread(file_path)
    # todo 简单实现一下
    cv2.imwrite(file_path.replace(".tif", ".jpg"), img)


# 为了加快转换的速度，去掉了一个gamma函数
# 还有使用列表来加快计算的方法，这边没有使用
# 有两种矩阵，虽然只有一点差别，但姑且记录下来
# M = np.array([[0.4124, 0.3576, 0.1805],
#               [0.2126, 0.7152, 0.0722],
#               [0.0193, 0.1192, 0.9505]])
M = np.array([[0.412453, 0.357580, 0.180423],
              [0.212671, 0.715160, 0.072169],
              [0.019334, 0.119193, 0.950227]])


def f_func(im_channel):
    # im_channel取值范围：[0,1]
    return np.power(im_channel, 1 / 3) if im_channel > 0.008856 else 7.787 * im_channel + 0.137931


def anti_f_func(im_channel):
    return np.power(im_channel, 3) if im_channel > 0.206893 else (im_channel - 0.137931) / 7.787


# 像素值RGB转XYZ空间，pixel格式:(B,G,R)
# 返回XYZ空间下的值
def rgb2xyz(rgb):
    """
    :param rgb: 数据图像格式是rgb
    :return:
    """
    rgb = np.array([rgb[0], rgb[1], rgb[2]])
    xyz = np.dot(M, rgb.T)
    xyz = xyz / 255.0
    return xyz[0] / 0.95047, xyz[1] / 1.0, xyz[2] / 1.08883


def xyz2lab(xyz):
    """
    :param xyz: 像素xyz空间下的值
    :return: 返回Lab空间下的值
    """
    xyz = [f_func(x) for x in xyz]
    L = 116 * xyz[1] - 16 if xyz[1] > 0.008856 else 903.3 * xyz[1]
    a = 500 * (xyz[0] - xyz[1])
    b = 200 * (xyz[1] - xyz[2])
    return L, a, b


def rgb2lab(pixel):
    """
    RGB空间转Lab空间
    :param pixel: RGB空间像素值，格式：[G,B,R]
    :return: 返回Lab空间下的值
    """
    return xyz2lab(rgb2xyz(pixel))


def lab2xyz(Lab):
    fY = (Lab[0] + 16.0) / 116.0
    fX = Lab[1] / 500.0 + fY
    fZ = fY - Lab[2] / 200.0

    x = anti_f_func(fX)
    y = anti_f_func(fY)
    z = anti_f_func(fZ)

    x = x * 0.95047
    y = y * 1.0
    z = z * 1.0883

    return x, y, z


def xyz2rgb(xyz):
    xyz = np.array(xyz)
    xyz = xyz * 255
    rgb = np.dot(np.linalg.inv(M), xyz.T)
    # rgb = rgb * 255
    rgb = np.uint8(np.clip(rgb, 0, 255))
    return rgb


def Lab2RGB(lab):
    return xyz2rgb(lab2xyz(lab))
