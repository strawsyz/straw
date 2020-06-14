import cv2
import numpy as np
from scipy import ndimage


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
            roi_gray = gray[y:y + h, x:x + h]
            # 检测眼睛
            # 因为眼睛是比较小的人脸特征，所以通过限制眼睛的最小尺寸（40,40）来去掉假阳性
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.03, 5, 0, (20, 20))
            for (ex, ey, ew, eh) in eyes:
                # 画出眼睛的矩形
                cv2.rectangle(img, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        cv2.imshow("camera", frame)
        if cv2.waitKey(1000 // 12) & 0xff == ord("q"):
            break
    camera.release()
    cv2.destroyAllWindows()


def detect_face(file_name):
    # face_cascade负责人脸检测
    face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
    # 读取图像转为灰度图像，人脸检测一般用灰度图像
    img = cv2.imread(file_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 进行人脸检测
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        # 画出检测的人脸
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return img


def detect_circle(img):
    """圆检测
    cv2.HoughCircles函数用于检测圆"""
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_temp = cv2.medianBlur(gray_img, 5)
    circles = cv2.HoughCircles(img_temp, cv2.HOUGH_GRADIENT, 1, 120, param1=100, param2=30,
                               minRadius=0, maxRadius=0)
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # 画圆形
        cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # 画圆心
        cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
    return img


def detect_line(img, threshold1=50, threshold2=120, min_line_length=20, max_line_gap=5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 使用canny滤波器去噪，留下边缘
    edges = cv2.Canny(gray, threshold1, threshold2)
    # 检测直线
    # 第一个参数 待处理的图像
    # 第二、三个参数是线段的集合表示rho，theta，一般分别取1和np.pi/180
    # 第四参数 阈值。低于该阈值的直线会被忽略
    # 第五参数 min_line_length 最小直线长度
    # 第六参数 max_line_gap 最大直线间隔
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, min_line_length, max_line_gap)
    for [[x1, y1, x2, y2]] in lines:
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return img


def threshold(img, thresh=127, maxval=255, type=0):
    # ret 可能是二值法的分界线
    ret, thresh = cv2.threshold(img, thresh, maxval, type)
    return thresh


def draw_contours_gray(img4find_contours, img4show=None, contours_color=(0, 255, 0), thickness=2):
    # 返回 图像的轮廓 以及 他们的层次 cv2.RETR_TREE会得到图片的整体轮廓
    # todo 待理解
    _, contours, hierarchy = cv2.findContours(img4find_contours, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if img4show is None:
        img4show = img4find_contours
    color_img = cv2.cvtColor(img4show, cv2.COLOR_GRAY2BGR)
    # 使用轮廓将图像的轮廓画成绿色
    return cv2.drawContours(img4show, contours, -1, contours_color, thickness)


def convolve_gray(img, kernel=[[-1, -1, -1],
                               [-1, 8, -1],
                               [-1, -1, -1]]):
    """
    对图像卷积，输入图像必须是灰度图像
    :param img:
    :param kernel:
    :return:
    """
    # img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)  # 读取灰度值
    return ndimage.convolve(img, kernel)


def low_pass_filter(img, ksize=(11, 11), sigmaX=0):
    # todo 还需要理解
    blurred = cv2.GaussianBlur(img, ksize, sigmaX)
    # 与原始图像计算差值
    return img - blurred


def canny_gray(img, threshold1=200, threshold2=300):
    """貌似只能处理灰度图"""
    # todo 还需要理解
    # 使用canny边缘检测算法
    return cv2.Canny(img, threshold1=threshold1, threshold2=threshold2)


def canny_threshold(low_threshold):
    detected_edges = cv2.GaussianBlur(gray, (3, 3), 0)
    detected_edges = cv2.Canny(detected_edges, low_threshold, low_threshold * ratio, apertureSize=kernel_size)
    dst = cv2.bitwise_and(img, img, mask=detected_edges)  # just add some colours to edges from original image.
    cv2.imshow('canny demo', dst)


if __name__ == '__main__':

    lowThreshold = 0
    max_lowThreshold = 100
    ratio = 3
    kernel_size = 3
    path = "C:\\Users\Administrator\PycharmProjects\straw\my_cv\polyp\source\Proc201506160021_1_1.png"
    # path = "C:\\Users\Administrator\PycharmProjects\straw\my_cv\polyp\source\\06.jpg"
    img = cv2.imread(path)
    img = cv2.resize(img, (224, 224))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.namedWindow('canny')

    cv2.createTrackbar('Min threshold', 'canny', lowThreshold, max_lowThreshold, canny_threshold)

    canny_threshold(0)  # initialization
    if cv2.waitKey(0) == 27:
        cv2.destroyAllWindows()
        # edges = cv2.Canny(np.array([[0, 1234], [1234, 2345]], dtype=np.uint16), 50, 100)
        # canny_gray("source/Proc201506020034_1_2.png")
