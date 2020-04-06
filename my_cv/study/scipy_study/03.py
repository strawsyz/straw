from pylab import *
from PIL import Image
import numpy as np
from scipy.ndimage import measurements, morphology

# 形态学（或数学形态学）是度量和分析基本形状的图像处理方法的基本框架与集合。
# 形态学通常用于处理二值图像，但是也能够用于灰度图像。

# 载入图像，然后使用阈值化操作，以保证处理的图像为二值图像
im = array(Image.open('test.jpg').convert('L'))
# 通过和 1 相乘，脚本将布尔数组转换成二进制表示。
im = 1 * (im < 240)
# 使用 label() 函数寻找单个的物体，
# 并且按照它们属于哪个对象将整数标签给像素赋值。
labels, nbr_objects = measurements.label(im)
imshow(labels)
# nbr_objects是计算出的物体的数目
print("Number of objects:", nbr_objects)
# 形态学二进制开（binary open）操作更好地分离各个对象
# binary_opening() 函数的第二个参数指定一个数组结构元素。
# 该数组表示以一个像素为中心时，使用哪些相邻像素。
# 在这种情况下，我们在 y 方向上使用 9 个像素
# （上面 4 个像素、像素本身、下面 4 个像素），
# 在 x 方向上使用 5 个像素。你可以指定任意数组为结构元素，
# 数组中的非零元素决定使用哪些相邻像素。
# 参数 iterations 决定执行该操作的次数。
im_open = morphology.binary_opening(im, ones((9, 5)), iterations=2)

labels_open, nbr_objects_open = measurements.label(im_open)
figure()
imshow(labels)
print("Number of objects:", nbr_objects_open)
show()
