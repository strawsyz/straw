import cv2

# cv2.approxPolyDP(curve, epsilon, closed)
# 该函数用来计算近视的多边形框
# 第一个参数为 轮廓
# 第二个参数为 e值 ，表示源轮廓与近似多边形的周长之间的最大差值
# 该差值越小，近似多边形与源轮廓就越相似
# 第三个参数为 “布尔标记”，表示这个多边形是否闭合
# epsilon = 0.01 * cv2.arcLength(cnt, True)  得到轮廓的周长信息作为参考值
# approx = cv2.approxPolyDP(cnt, epsilon, True)  得到近似多边形框
# hull = cv2.convexHull(cnt)  绘制多边形的凸包，获取处理过的轮廓信息
