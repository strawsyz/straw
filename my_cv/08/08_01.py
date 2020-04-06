import cv2
import numpy as np

"""基本的运动检测
通过计算帧之间的差异
需要使用使用第一帧作为背景"""

camera = cv2.VideoCapture(0)

es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 4))

kernel = np.ones((5, 5), np.uint8)
background = None

while True:
    ret, frame = camera.read()
    # 将第一帧设为背景
    if background is None:
        background = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 因为输入的视频会因震动、光照变化等原因而产生噪音。
        # 对噪声进行平滑，是为了避免在运动和跟踪是检测出来
        background = cv2.GaussianBlur(background, (21, 21), 0)
        continue

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)
    # 计算与背景帧的差异，得到一个差分图。
    diff = cv2.absdiff(background, gray_frame)
    # 使用阈值的到一个黑白图像
    diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
    # 膨胀（dilate）图像，从而对孔（hole）和缺陷（imperfection）进行归一化处理
    diff = cv2.dilate(diff, es, iterations=2)
    # 计算图像中的目标轮廓
    image, cnts, hierarchy = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts:
        # 对于光照不变和噪声低的摄像头，可以不设定轮廓最小尺寸的阈值
        if cv2.contourArea(c) < 1500:
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("contours", frame)
    cv2.imshow("diff", diff)
    if cv2.waitKey(1000 // 12) & 0xff == ord("q"):
        break
cv2.destroyAllWindows()
camera.release()
