import numpy as np
import cv2
from matplotlib import pyplot as plt

"""
使用orb和暴力匹配算法来匹配两张图片
可以找两张相似的图片来匹配"""
# 读取图片为灰度图片
img1 = cv2.imread('test.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('corner.png', cv2.IMREAD_GRAYSCALE)

orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:40], img2, flags=2)

plt.imshow(img3)
plt.show()
