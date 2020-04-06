import cv2

"""建立一个窗口捕捉摄像头显示的内容
只有当左键点击窗口，然后按任意键盘键，才会退出窗口"""
clicked = False


def on_mouse(event, x, y, flags, param):
    global clicked
    if event == cv2.EVENT_LBUTTONUP:
        clicked = True


camera_capture = cv2.VideoCapture(0)
cv2.namedWindow('test')
cv2.setMouseCallback('test', on_mouse)

success, frame = camera_capture.read()
# cv2.waitKey(1) 参数表示等待键盘触发的时间，返回值为-1表示没有见按下
while success and cv2.waitKey(1) == -1 and not clicked:
    cv2.imshow('test', frame)
    success, frame = camera_capture.read()
cv2.destroyAllWindows()
camera_capture.release()
