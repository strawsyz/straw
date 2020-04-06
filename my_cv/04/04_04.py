import numpy as np
import cv2

from matplotlib import pyplot as plt
"""使用分水岭的方法来分割图像"""
img = cv2.imread('test.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 为图像设置阈值，将图像分为黑色和白色部分
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# 使用morphologyEx变化对图形进行膨胀操作
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
# 得到大部分都是背景的区域
sure_bg = cv2.dilate(opening, kernel, iterations=3)
# 通过distanceTransform来获取确定的前景区域
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
# 应用一个阈值来决定哪些区域是前景
ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
# 拿出前景和背景重合的区域
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)
# 设定栅栏来阻止水的汇聚
ret, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
# 将重合区域设为0
markers[unknown == 255] = 0
# 将水放入
markers = cv2.watershed(img, markers)
# 将栅栏部分设为红色
img[markers == -1] = [255, 0, 0]
plt.imshow(img)
plt.show()
