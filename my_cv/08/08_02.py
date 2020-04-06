import cv2
"""使用MOG2进行背景分割的简单例子
BackgroundSubtractor类是专门针对视频的，会对每帧的环境进行学习
另外，它能计算阴影。可以通过检测阴影，排除图像的阴影区域
效果有点帅"""
cap = cv2.VideoCapture(0)

mog = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    fgmask = mog.apply(frame)
    cv2.imshow('frame', fgmask)
    if cv2.waitKey(30) & 0xff == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
