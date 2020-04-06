import cv2
import numpy as np

# 模糊并下采样
img = cv2.pyrDown(cv2.imread('test.jpg', cv2.IMREAD_UNCHANGED))
print(img.shape)
cv2.imshow('temp', img)
# 转为灰度图像，并执行二值化操作
ret, thresh = cv2.threshold(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)
# 找到轮廓
_, contours, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
img = cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
for c in contours:
    # 这边获得的轮廓并不是矩形，所以要先对图形进行处理
    # 计算简单的边界框
    x, y, w, h = cv2.boundingRect(c)
    # 画出该矩形
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # 找到最小区域
    rect = cv2.minAreaRect(c)
    # 获得区域的顶点坐标
    box = cv2.boxPoints(rect)
    # 将坐标转为整数
    box = np.int0(box)
    # 画出矩形，该函数会修改源图像
    # 第二参数接受保存轮廓的数据
    # 第三个参数要绘制的轮廓在数组中的索引
    # 第四个参数是颜色
    # 第5个参数是厚度
    cv2.drawContours(img, [box], 0, (0, 0, 255), 3)
    # 找到边界轮廓的最小闭圆
    (x, y), radius = cv2.minEnclosingCircle(c)

    center = (int(x), int(y))
    radius = int(radius)
    # 画圆
    img = cv2.circle(img, center, radius, (0, 255, 0), 2)

cv2.drawContours(img, contours, -1, (255, 0, 0), 1)
cv2.imshow('contours', img)

cv2.waitKey()
cv2.destroyAllWindows()
