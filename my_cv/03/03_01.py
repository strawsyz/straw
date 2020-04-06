import cv2
import numpy as np

from scipy import ndimage

kernel_3x3 = np.array([[-1, -1, -1],
                       [-1, 8, -1],
                       [-1, -1, -1]])
kernel_5x5 = np.array([[-1, -1, -1, -1, -1],
                       [-1, 1, 2, 1, -1],
                       [-1, 2, 4, 2, -1],
                       [-1, 1, 2, 1, -1],
                       [-1, -1, -1, -1, -1]])
# 读入图像，转为灰度格式
img = cv2.imread('test.jpg', 0)
# 进行卷积
# 方法一
k3 = ndimage.convolve(img, kernel_3x3)
# 方法二
k5 = ndimage.convolve(img, kernel_5x5)

# 第三种方法，该方法效果最好
# 对图像应用低通滤波器
blurred = cv2.GaussianBlur(img, (11, 11), 0)
# 与原始图像计算差值
g_hpf = img - blurred

cv2.imshow('3x3', k3)
cv2.imshow('5x5', k5)
cv2.imshow('g_hpf', g_hpf)
cv2.waitKey()
cv2.destroyAllWindows()
