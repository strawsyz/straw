from PIL import Image
from pylab import *


# 图像平均操作是减少图像噪声的一种简单方式，
# 通常用于艺术特效。
def compute_average(im_list):
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
          print (im_name + '...skipped')
    averageim /= len(im_list)

    # 返回uint8 类型的平均图像
    return array(averageim, 'uint8')
