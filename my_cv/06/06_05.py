import numpy as np
import cv2
from matplotlib import pyplot as plt

"""使用FLANN来匹配
FLANN是近似最近邻的快速库，可以在项目中自由使用"""
query_image = cv2.imread('test.jpg', cv2.IMREAD_GRAYSCALE)
training_image = cv2.imread('corner.png', cv2.IMREAD_GRAYSCALE)

# 创建SIFT
sift = cv2.xfeatures2d.SIFT_create()
# 检测并计算
kp1, des1 = sift.detectAndCompute(query_image, None)
kp2, des2 = sift.detectAndCompute(training_image, None)
# FLANN 匹配器的参数
FLANN_INDEX_KDTREE = 0
# algorithm有LinearIndex、KTreeIndex、KMeansIndex、CompositeIndex和AutotuneIbdex
# KTreeIndex非常灵活，trees是密度树的数量，理想值的数量在1~16之间
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
# search_params中的字段checks，用来指定索引树要被遍历的次数，该值越高，计算的时间越长，但也会越准确
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1, des2, k=2)

matches_mask = [[0, 0] for i in range(len(matches))]

# 丢弃距离大于0.7的匹配
for i, (m, n) in enumerate(matches):
    if m.distance < 0.7 * n.distance:
        matches_mask[i] = [1, 0]

draw_params = dict(matchColor=(0, 255, 0), singlePointColor=(255, 0, 0),
                   matchesMask=matches_mask, flags=0)

result_image = cv2.drawMatchesKnn(query_image, kp1, training_image, kp2, matches, None, **draw_params)
plt.imshow(result_image)
plt.show()
