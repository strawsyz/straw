'''对应第二章中的小例子'''
import numpy as np
import cv2

img = np.zeros((3, 3), dtype=np.uint8)
print(img)
print(img.shape)

img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
print(img)
print(img.shape)

print("==============")
print("convert png to jpg")
# imread 函数会删除所有alpha通道的信息（透明度）
image = cv2.imread('test.png')
cv2.imwrite('test_png_2_jpg.jpg', image)
# imwrite 函数要求图像为BGR或灰度格式，并且每个通道要有一定的位
gray_image = cv2.imread('test.png', cv2.IMREAD_GRAYSCALE)
cv2.imwrite('test_rgb_2_gray.png', gray_image)

# image[y坐标或行（0表示顶部）,x坐标或列（0在最左边）,颜色通道]

# 设置像素值,对单像素的操作，第二种表示方式更有效
# 一、 image[0, 0] = 128
# 二、 image.setitem((0,0). 128) 取值 image.item((0,0))

# 图像转为一维数组
# byte_array = bytearray(image)
# 将一维数组转为图像
# gray_image = numpy.array(grayByteArray).reshape(height, width)
# bgr_image = numpy.array(bgrByteArray).reshape(height, width,3)
