import cv2

"""从摄像机检测人脸，将检测到的人脸保存起来转为200x200的灰度图片
如果不设置保存多少张图片就会一直不断的保存下去
会有一些很模糊的图像 
点击q键退出"""
def generate(face_num=float('inf')):
    face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
    # eye_cascade = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')

    camera = cv2.VideoCapture(0)
    count = 0

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
            f = cv2.resize(gray[y:y + h, x:x + w], (200, 200))
            # 将检测到的人脸保存起来
            cv2.imwrite('{}.pgm'.format(str(count)), f)
            count += 1
        cv2.imshow("camera", frame)
        if count > face_num:
            break
        if cv2.waitKey(1000//12) & 0xff ==ord("q"):
            break
    camera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    generate(100)