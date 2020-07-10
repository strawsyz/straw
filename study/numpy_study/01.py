import numpy as np
from PIL import Image
from pylab import *

im = array(Image.open('test.jpg'))
print(im.shape, im.dtype)

im = array(Image.open('test.jpg').convert('L'), 'f')
print(im.shape, im.dtype)

# 建一个全黑的画布
im = np.zeros((1080, 1920, 3))
# 设定值
im[1:100, 0:200, 0:3] = 100
# 求和
print(im[:100, :50].sum())
# 计算平均值
print(im[22].mean())
print(im.shape)
imshow(im)
show()
