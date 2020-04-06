import numpy as np
import cv2
"""使用均值漂移检测目标移动的例子
效果很不好
这种方式存在一个问题，就是窗口的大小不与跟踪帧中的目标大小一起变化
"""
cap = cv2.VideoCapture(0)
# 获得第一帧图像
ret, frame = cap.read()
# 标志 ROI的区域
r, h, c, w = 10, 200, 10, 200
track_window = (c, r, w, h)

# 提取roi区域
roi = frame[r:r + h, c:c + w]
# 将图片转为HSV格式
hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
# 创建包含具有HSV值ROI所有像素的掩码，HSV值的范围在上下界之间
mask = cv2.inRange(hsv_roi, np.array((100., 30., 32.)), np.array((180., 120., 255.)))
# 计算roi的直方图
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
# 将值归一化到0~255的范围之内
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# 均值漂移在收敛之前会迭代多次，但不一定能保证收敛
# 下面是均值漂移终止的条件
# 均值漂移迭代10次或者中心移动至少1个像素
# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while True:
    ret, frame = cap.read()
    if ret == True:
        # 将图片转化为HSV色彩空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # 执行直方图反向投影
        # cv2.calcBackProject用来计算每个像素属于原始图像的概率
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        print(dst)
        # 获得跟踪目标的新位置
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)

        # 计算窗口的新坐标。将位置的矩形画出来
        x, y, w, h = track_window
        img = cv2.rectangle(frame, (x, y), (x + w, y + h), 255, 2)
        cv2.imshow('img', img)

        k = cv2.waitKey(60) & 0xff
        if k == ord("q"):
            break
    else:
        break

cv2.destroyAllWindows()
cap.release()
