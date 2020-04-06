import numpy as np
import cv2
from matplotlib import pyplot as plt
"""使用kNN算法来进行匹配
区别：match函数返回最佳匹配，knnMatch返回k个匹配，开发人员可以进一步处理这些匹配
例如：可以遍历匹配，采用比率测试来过滤掉不满足用户定义条件的匹配"""
# 读取图片为灰度图片
img1 = cv2.imread('test.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('corner.png', cv2.IMREAD_GRAYSCALE)

orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# 书上代码中 k=2 会报错，改为1不会报错
matches = bf.knnMatch(des1, des2, k=1)
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, img2, flags=2)
plt.imshow(img3)
plt.show()
