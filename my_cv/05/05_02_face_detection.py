import cv2

"""
用摄像头检测正脸
总是把鼻孔当做眼睛。。。
点击q退出窗口"""


def detect_face_eye():
    face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')
    camera = cv2.VideoCapture(0)

    while True:
        # 捕获帧
        ret, frame = camera.read()
        # 将捕获的帧转化为灰度图像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 检测人脸
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            # 画出人脸的框
            img = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # 取出灰度图像上检测出来的脸的部分
            roi_gray = gray[y:y + h, x:x + h]
            # 检测眼睛
            # 因为眼睛是比较小的人脸特征，所以通过限制眼睛的最小尺寸（40,40）来去掉假阳性
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.03, 5, 0, (20, 40))
            for (ex, ey, ew, eh) in eyes:
                # 画出眼睛的矩形
                cv2.rectangle(img, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)
        cv2.imshow("camera", frame)
        if cv2.waitKey(1000 // 12) & 0xff == ord("q"):
            break
    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    detect_face_eye()
