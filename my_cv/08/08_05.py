import cv2
import numpy as np

"""使用CAMShift检测目标移动的例子
代码与08_04.py相似
"""

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
# 标记感兴趣的区域
r, h, c, w = 10, 200, 10, 200
track_window = (c, r, w, h)
# 提取ROI将其转换为HSV色彩空间
roi = frame[r:r + h, c:c + w]
hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
# 创建包含具有HSV值ROI所有像素的掩码，HSV值的范围在上下界之间
mask = cv2.inRange(hsv_roi, np.array((100., 30., 32.)), np.array((180., 120., 255.)))
# 计算ROI的直方图
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
# 将值归一化到0~255的范围之内
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
# 均值漂移终止条件
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while True:
    ret, frame = cap.read()
    if ret == True:
        # 将图片转化为HSV色彩空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # 执行直方图反向投影
        # cv2.calcBackProject用来计算每个像素属于原始图像的概率
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        # 获得跟踪目标的新位置
        ret, track_window = cv2.CamShift(dst, track_window, term_crit)
        print(ret)
        # 获得4个坐标点
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        print(pts)
        img = cv2.polylines(frame, [pts], True, 255, 2)
        # 计算窗口的新坐标。将位置的矩形画出来
        cv2.imshow('img', img)

        k = cv2.waitKey(60) & 0xff
        if k == ord("q"):
            break
    else:
        break

cv2.destroyAllWindows()
cap.release()
