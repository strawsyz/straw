import cv2
import numpy as np

"""轮廓检测"""
"""改了一下，忘记之前是什么样的了"""
# 在200x200的图像中央放置100x100的白色方块
img = np.zeros((200, 200), dtype=np.uint8)
img[50:150, 50:150] = 1

# 对图像进行二值化操作
ret, thresh = cv2.threshold(img, 100, 255, 0)
# 返回 图像的轮廓 以及 他们的层次 cv2.RETR_TREE会得到图片的整体轮廓
_, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# 使用轮廓将图像的轮廓画成绿色
img = cv2.drawContours(color_img, contours, -1, (0, 255, 0), 2)
cv2.imshow('contours', color_img)
cv2.imshow('thresh', thresh)
print(ret)
cv2.waitKey()
cv2.destroyAllWindows()
