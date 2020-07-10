import numpy as np
from PIL import Image
from pylab import *
from scipy.ndimage import filters

im = np.array(Image.open('test.jpg').convert('L'))

imx = np.zeros(im.shape)
# sobel() 函数的第二个参数表示选择 x 或者 y 方向导数，
# 第三个参数保存输出的变量
filters.sobel(im, 1, imx)

imy = np.zeros(im.shape)
filters.sobel(im, 0, imy)

magnitude = np.sqrt(imx ** 2 + imy ** 2)
# 正导数显示为亮的像素，负导数显示为暗的像素,。灰色区域表示导数的值接近于零。

Image.fromarray(magnitude).show()
# show()
print(magnitude)

# 使用高斯倒数滤波器
sigma = 5  # 标准差

imx = zeros(im.shape)
# 第三个参数指定对每个方向计算哪种类型的导数，第二个参数为使用的标准差
filters.gaussian_filter(im, (sigma, sigma), (0, 1), imx)

imy = zeros(im.shape)
filters.gaussian_filter(im, (sigma, sigma), (1, 0), imy)
magnitude = np.sqrt(imx ** 2 + imy ** 2)
imshow(magnitude)
show()
Image.fromarray(magnitude).show()
