import numpy as np
import cv2

from matplotlib import pyplot as plt
"""使用grabCut算法，区分前景和背景"""
# 加载图像，然后创建于图像同形状的掩模
img = cv2.imread('test.jpg')
mask = np.zeros(img.shape[:2], np.uint8)
# 使用一个矩形初始化
rect = (100, 50, 421, 378)
# 创建以0填充的前景和背景模型，要基于初始矩形所留下的区域来决定
bgd_model = np.zeros((1, 65), np.float64)
fgd_model = np.zeros((1, 65), np.float64)
# 使用指定的空模型和掩模来运行GrabCut算法
# 第6个参数是 算法的迭代次数，可以设得更大，但是像素分类会在某个地方收敛
# 这时在增加迭代次数也不会有改进
cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 1, cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
img = img * mask2[:, :, np.newaxis]
# 展示图片结果
plt.subplot(121), plt.imshow(img)
plt.title('grabcut'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(cv2.cvtColor(cv2.imread('test.jpg'), cv2.COLOR_BGR2RGB))
plt.title('original'), plt.xticks([]), plt.yticks([])
plt.show()
