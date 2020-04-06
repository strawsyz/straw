import cv2
import numpy as np
"""使用cornerHarris检测图像角点"""
img = cv2.imread('corner.png')
# 将图像转化为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
# 检测图像角点
# 第二个参数值越小，标记角点的记号越小
# 第三个参数限定了Sobel算子的中孔（aperture）
# 该参数定义了角点检测的敏感度，取值必须在3到31之间的奇数
dst = cv2.cornerHarris(gray, 2, 23, 0.04)
# 将角点的位置标记为红色
img[dst > 0.01 * dst.max()] = [0, 0, 255]
while True:
    cv2.imshow('corners', img)
    if cv2.waitKey(1000//12) & 0xff == ord('q'):
        break
cv2.destroyAllWindows()
