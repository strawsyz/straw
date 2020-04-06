import cv2
"""
1、使用高斯滤波器对图像去噪
2、计算梯度
3、在边缘上使用非最大抑制（NMS）
4、在检测到的边缘上使用双阈值去除假阳性
5、最后分析所有的边缘及其之间的连接，保留真的边缘，消除不明显的边缘"""
# 转化为灰度图像
img = cv2.imread('test.jpg', 0)
# 使用canny边缘检测算法
cv2.imwrite('test_canny.jpg', cv2.Canny(img, 200, 300))
cv2.imshow("canny", cv2.imread('test_canny.jpg'))
cv2.waitKey()
cv2.destroyAllWindows()
