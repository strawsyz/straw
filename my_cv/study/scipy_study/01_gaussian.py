from PIL import Image
import numpy as np
from scipy.ndimage import filters

# 关于scipy
# http://docs.scipy.org/doc/scipy/reference/ndimage.html
# 高斯模糊
# 图像的高斯模糊是非常经典的图像卷积例子。
# 本质上，图像模糊就是将（灰度）图像 I 和一个高斯核进行卷积操作：

im = np.array(Image.open('test.jpg').convert('L'))
# guassian_filter() 函数的最后一个参数表示标准差。
im2 = filters.gaussian_filter(im, 5)
im2 = np.array(im2, 'uint8')
# 随着 σ 的增加，一幅图像被模糊的程度。σ 越大，处理后的图像细节丢失越多。

# 如果打算模糊一幅彩色图像，只需简单地对每一个颜色通道进行高斯模糊：
im = np.array(Image.open('test.jpg'))
im2 = np.zeros(im.shape)
for i in range(3):
    im2[:, :, i] = filters.gaussian_filter(im[:, :, i], 5)
im2 = np.uint8(im2)
Image.fromarray(im2).show()
