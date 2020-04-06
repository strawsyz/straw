import cv2
import numpy as np

"""使用卡尔曼滤波器跟踪鼠标的例子"""

# 创建大小为800x800的空帧
frame = np.zeros((800, 800, 3), np.uint8)
# 初始化测量坐标和鼠标运动预测的数组
last_measurement = current_measurement = np.array((2, 1), np.float32)
last_prediction = current_prediction = np.zeros((2, 1), np.float32)


def mousemove(event, x, y, s, p):
    global frame, current_measurement, measurement, last_measurement, \
        current_prediction, last_prediction

    last_prediction = current_prediction
    last_measurement = current_measurement

    current_measurement = np.array([[np.float32(x)], [np.float32(y)]])
    # 用当前测量来校正卡尔曼滤波器
    kalman.correct(current_measurement)
    # 计算卡尔曼的预测值
    current_prediction = kalman.predict()
    lmx, lmy = last_measurement[0], last_measurement[1]
    cmx, cmy = current_measurement[0], current_measurement[1]
    lpx, lpy = last_prediction[0], last_prediction[1]
    cpx, cpy = current_prediction[0], current_prediction[1]
    # 绘制上一次测量到当前测量以及上一次预测到当前预测的两条线
    cv2.line(frame, (lmx, lmy), (cmx, cmy), (0, 100, 0))
    cv2.line(frame, (lpx, lpy), (cpx, cpy), (0, 0, 200))


cv2.namedWindow('kalman_tracker')
cv2.setMouseCallback('kalman_tracker', mousemove)
# 卡尔曼滤波器的构造函数有如下可选参数
# dynamParams:表示状态的维度
# measureParams:表示测量的维度
# controlParams:表示控制的维度
# vector.type: 表示所创建的矩阵类型，可为（CV_32F或CV_64F）
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03

while True:
    cv2.imshow("kalman_tracker", frame)
    if cv2.waitKey(30) & 0xff == ord("q"):
        break
cv2.destroyAllWindows()

