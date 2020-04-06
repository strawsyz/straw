import cv2
import numpy as np

"""直线检测
通过 cv2.HoughLinesP 和 cv2.HoughLines函数来完成
cv2.HoughLines 使用标准的Hough变换，cv2.HoughLinesP使用概率Hough变换
cv2.HoughLinesP的计算代价会少一点，速度更快
"""
img = cv2.imread('test.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 使用canny滤波器去噪，留下边缘
edges = cv2.Canny(gray, 50, 120)
min_line_length = 20
max_line_gap = 5
# 检测直线
# 第一个参数 待处理的图像
# 第二、三个参数是线段的集合表示rho，theta，一般分别取1和np.pi/180
# 第四参数 阈值。低于该阈值的直线会被忽略
# 第五参数 min_line_length 最小直线长度
# 第六参数 max_line_gap 最大直线间隔
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, min_line_length, max_line_gap)
for [[x1, y1, x2, y2]] in lines:
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow("edges", edges)
cv2.imshow('lines', img)
cv2.waitKey()
cv2.destroyAllWindows()
