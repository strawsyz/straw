from PIL import Image
from pylab import *
import numpy as np


def imresize(im, size):
    """
     使用PIL 对象resize图像数组
    :param im: 图像数组
    :param size: 重新定义的大小
    :return: 
    """
    pil_im = Image.fromarray(np.uint8(im))

    return array(pil_im.resize(size))


# 直方图均衡化是指将一幅图像的灰度直方图变平，
# 使变换后的图像中每个灰度值的分布概率都相同。
def histeq(im, nbr_bins=256):
    """
    对灰度图像进行直方图均衡化
    :param im:灰度图像数组
    :param nbr_bins:直方图中使用小区间的数目
    :return:直方图均衡化后的图像，用来做像素值映射的累积分布函数
    """
    imhist, bins = histogram(im.flatten(), nbr_bins, normed=True)
    # cumulative distribution function
    cdf = imhist.cumsum()
    # 归一化，使范围在0~255
    cdf = 255 * cdf / cdf[-1]
    # 使用累积分布函数的线性插值，计算新的像素值
    im2 = interp(im.flatten(), bins[:-1], cdf)
    return im2.reshape(im.shape), cdf

if __name__ == '__main__':
    im = array(Image.open('test.jpg').convert('L'))
    im2, cdf = histeq(im)
    imshow(im)
    figure()
    imshow(im2)
    figure()
    hist(im.flatten(), 128)
    figure()
    hist(im2.flatten(), 128)
    show()
    Image.fromarray(im).save('test_gray.jpg')
    im2 = np.uint8(im2)
    Image.fromarray(im2).save('test_gray_histeq.jpg')
