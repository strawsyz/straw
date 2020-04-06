import cv2

"""基础的人脸检测
可以检测出test.jpg
无法检测test.png"""
file_name = 'test.jpg'

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
    # cv2.namedWindow('detected!!')
    cv2.imshow('detected!!', img)
    cv2.imwrite('detected.jpg', img)
    cv2.waitKey(0)


detect_face(file_name)
