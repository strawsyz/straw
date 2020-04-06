import numpy as np
import cv2
from matplotlib import pyplot as plt

"""显示图像的单应性
单应性是一个条件，该条件表明当两幅图像出现投影畸变时，他们还能彼此匹配 """
# 最少匹配数
MIN_MATCH_COUNT = 10

img1 = cv2.imread('test.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('corner.png', cv2.IMREAD_GRAYSCALE)

sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1, des2, k=2)

good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)

if len(good) > MIN_MATCH_COUNT:
    # 在原始图像和训练图像中发现关键点
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    # 发现单应性
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matches_mask = mask.ravel().tolist()

    # 第二张图像计算相对于原始目标的投影畸变，并绘制边框
    h, w = img1.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)
    img2 = cv2.ploylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
else:
    print('没有找到足够的匹配 - %d/%d' % (len(good), MIN_MATCH_COUNT))
    matches_mask = None

draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None,
                   matchesMask=matches_mask, flags=0)

img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)

plt.imshow(img3,'gray')
plt.show()