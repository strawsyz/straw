import cv2
import numpy as np
import os.path as path
import argparse

"""
检查行人移动的应用，用kalman预测人的移动的效果不佳
尤其是人与人重叠的时候，就会导致误判
一、检查第一帧
二、检查后面输入的帧，从场景的开始通过背景分割器来识别场景中的行人
三、为每个人行人建立ROI，并利用Kalman/CAMShift来跟踪行人ID
四、检查下一帧是否有进入场景的新行人"""

parser = argparse.ArgumentParser()
parser.add_argument("-a", "--algorithm", default="m",
                    help="m (or nothing) for meanShift and c for camshift")
args = vars(parser.parse_args())


def center(points):
    """calculates centroid of a given matrix"""
    x = (points[0][0] + points[1][0] + points[2][0] + points[3][0]) / 4
    y = (points[0][1] + points[1][1] + points[2][1] + points[3][1]) / 4
    return np.array([np.float32(x), np.float32(y)], np.float32)


font = cv2.FONT_HERSHEY_SIMPLEX


class Pedestrian():
    """行人类

    每个行人由一个ROI、一个Id，一个Kalman过滤器组成
    """

    def __init__(self, id, frame, track_window):
        """init the pedestrian object with track window coordinates"""
        self.id = int(id)
        # 设置ROI
        x, y, w, h = track_window
        self.track_window = track_window
        self.roi = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2HSV)
        roi_hist = cv2.calcHist([self.roi], [0], None, [16], [0, 180])
        self.roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

        # 设置卡尔曼滤波器
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                                               np.float32) * 0.03

        self.measurement = np.array((2, 1), np.float32)
        self.prediction = np.zeros((2, 1), np.float32)
        self.term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        self.center = None
        self.update(frame)

    def __del__(self):
        print("行人 %d 删除" % self.id)

    def update(self, frame):
        # 转化为HSV格式，用于计算行人HSV直方图
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        back_project = cv2.calcBackProject([hsv], [0], self.roi_hist, [0, 180], 1)
        # 使用CAMShift或均值漂移来跟踪行人的运动
        if args.get("algorithm") == "c":
            ret, self.track_window = cv2.CamShift(back_project, self.track_window, self.term_crit)
            pts = cv2.boxPoints(ret)
            pts = np.int0(pts)
            self.center = center(pts)
            cv2.polylines(frame, [pts], True, 255, 1)
        # if not args.get("algorithm") or args.get("algorithm") == "m":
        if args.get("algorithm") == "m":
            ret, self.track_window = cv2.meanShift(back_project, self.track_window, self.term_crit)
            x, y, w, h = self.track_window
            self.center = center([[x, y], [x + w, y], [x, y + h], [x + w, y + h]])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
        # 根据行人的位置校正正卡尔曼滤波器
        self.kalman.correct(self.center)
        prediction = self.kalman.predict()
        # 画出预测的位置的点
        cv2.circle(frame, (int(prediction[0]), int(prediction[1])), 4, (255, 0, 0), -1)
        # 在图像左上角打印行人信息
        # 增加阴影，防止因文字和图片颜色一致，而导致看不到图片
        cv2.putText(frame, "ID: %d -> %s" % (self.id, self.center), (11, (self.id + 1) * 25 + 1),
                    font, 0.6,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA)
        # actual info
        cv2.putText(frame, "ID: %d -> %s" % (self.id, self.center), (10, (self.id + 1) * 25),
                    font, 0.6,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA)


def main():
    # camera = cv2.VideoCapture(path.join(path.dirname(__file__), "traffic.flv"))
    camera = cv2.VideoCapture(path.join(path.dirname(__file__), "768x576.avi"))
    # camera = cv2.VideoCapture(path.join(path.dirname(__file__), "..", "movie.mpg"))
    # camera = cv2.VideoCapture(0)
    # 设置前20帧作为影响背景模型的帧
    history = 20
    # KNN 背景分割器
    bs = cv2.createBackgroundSubtractorKNN(history=history, detectShadows=True)

    # MOG 背景分割器
    # bs = cv2.bgsegm.createBackgroundSubtractorMOG(history = history)
    # bs.setHistory(history)

    # GMG 背景分割器
    # bs = cv2.bgsegm.createBackgroundSubtractorGMG(initializationFrames = history)

    cv2.namedWindow("surveillance")
    # 行人字典
    pedestrians = {}
    first_frame = True
    # 帧计数器
    frames_counter = 0
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
    while True:
        print(" -------------------- FRAME %d --------------------" % frames_counter)
        grabbed, frane = camera.read()
        if grabbed is False:
            print("failed to grab frame.")
            break

        ret, frame = camera.read()
        # 只需将帧传到背景分割器中
        fgmask = bs.apply(frame)

        # 给背景分割器几个帧作为历史帧
        if frames_counter < history:
            frames_counter += 1
            continue
        # 通过对前景掩模才用膨胀和腐蚀的方法来识别斑点和周围边框，找到帧中运动的对象
        th = cv2.threshold(fgmask.copy(), 127, 255, cv2.THRESH_BINARY)[1]
        th = cv2.erode(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
        dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 3)), iterations=2)
        image, contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        counter = 0
        for c in contours:
            if cv2.contourArea(c) > 500:
                (x, y, w, h) = cv2.boundingRect(c)
                # 用绿框画出行人实际位置
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                # 只给在第一帧中检测到的行人创建对应的对象
                if first_frame is True:
                    pedestrians[counter] = Pedestrian(counter, frame, (x, y, w, h))
                counter += 1

        for i, p in pedestrians.items():
            p.update(frame)

        # 设为False，让程序只跟踪第一帧中出现的行人
        first_frame = False
        frames_counter += 1

        cv2.imshow("surveillance", frame)
        # out.write(frame)
        if cv2.waitKey(110) & 0xff == ord("q"):
            break
    # out.release()
    camera.release()


if __name__ == "__main__":
    main()
