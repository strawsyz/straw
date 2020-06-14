import numpy as np


def create_dark(size=(1080, 1920, 3)):
    return np.zeros(size)


def reverse_color(img):
    return 255 - img


def standard(img, min, max):
    """标准化"""
    return (max - min) * img / 255 + min


# 直方图均衡化是指将一幅图像的灰度直方图变平，
# 使变换后的图像中每个灰度值的分布概率都相同。
def histeq(im, nbr_bins=256):
    """
    对灰度图像进行直方图均衡化
    :param im:灰度图像数组
    :param nbr_bins:直方图中使用小区间的数目
    :return:直方图均衡化后的图像，用来做像素值映射的累积分布函数
    """

    from pylab import *
    imhist, bins = histogram(im.flatten(), nbr_bins, normed=True)
    # cumulative distribution function
    cdf = imhist.cumsum()
    # 归一化，使范围在0~255
    cdf = 255 * cdf / cdf[-1]
    # 使用累积分布函数的线性插值，计算新的像素值
    im2 = interp(im.flatten(), bins[:-1], cdf)
    return im2.reshape(im.shape), cdf


def pca(x):
    """
    主成分分析
    :param x:矩阵X ，其中该矩阵中存储训练数据，每一行为一张图片的所有像素
    :return:投影矩阵（按照维度的重要性排序）、方差和均值
    """
    # 获取维数
    num_data, dim = x.shape

    # 数据中心化
    mean_x = x.mean(axis=0)
    x = x - mean_x

    if dim > num_data:
        # PCA- 使用紧致技巧
        M = np.dot(x, x.T)  # 协方差矩阵
        e, EV = np.linalg.eigh(M)  # 特征值和特征向量
        tmp = np.dot(x.T, EV).T  # 这就是紧致技巧
        V = tmp[::-1]  # 由于最后的特征向量是我们所需要的，所以需要将其逆转
        S = np.sqrt(e)[::-1]  # 由于特征值是按照递增顺序排列的，所以需要将其逆转
        for i in range(V.shape[1]):
            V[:, i] /= S
    else:
        # PCA- 使用SVD 方法
        U, S, V = np.linalg.svd(x)
        V = V[:num_data]  # 仅仅返回前nun_data 维的数据才合理

    # 返回投影矩阵、方差和均值
    return V, S, mean_x


def save(path, data, dtype='%i'):
    np.savetxt(path, data, dtype)


def load(path):
    return np.loadtxt(path)


# 图像平均操作是减少图像噪声的一种简单方式，
# 通常用于艺术特效。
def compute_average(im_list):
    from PIL import Image
    from pylab import *
    """
    计算图像列表的平均像素
    不使用mean（）函数减少内存占用
    需要所有图像的大小相同
    :param im_list: 图像路径列表
    :return: 平均后的图像
    """
    # 打开第一幅图像，将其存储在浮点型数组中
    averageim = array(Image.open(im_list[0]), 'f')
    for im_name in im_list[1:]:
        try:
            averageim += array(Image.open(im_name))
        except:
            print(im_name + '...skipped')
    averageim /= len(im_list)

    # 返回uint8 类型的平均图像
    return array(averageim, 'uint8')
