from PIL import Image
from pylab import *
import numpy as np

im = array(Image.open('test.jpg').convert('L'))
# 进行反相处理
im2 = 255 - im
# 将图像变换到100...200区间
im3 = (100.0 / 255) * im + 100
# 对图像像素值求平方后得到图像
im4 = 255.0 * (im / 255.0) ** 2
# 图像最小值
print(int(im3.min()))
# 图像最大值
print(int(im3.max()))
figure()
gray()
imshow(im2)
figure()
imshow(im3)
figure()
imshow(im4)
show()

pil_im = Image.fromarray(np.uint8(im3))
pil_im.save('test_100_2_200.jpg')