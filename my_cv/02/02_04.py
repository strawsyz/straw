import cv2

"""使用摄像机拍下图像保存在avi文件中"""
# 获得设备
camera_capture = cv2.VideoCapture(0)
# 猜想频率为60,设为30会有问题
time = 3  # 秒数
fps = 60
# fps = camera_capture.get(cv2.CAP_PROP_FPS)  直接获得频率的返回值为0.0
size = (int(camera_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(camera_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
video_writer = cv2.VideoWriter('MyCamera.avi', cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)

success, frame = camera_capture.read()
num_frames_remaining = time * fps - 1
while success and num_frames_remaining > 0:
    video_writer.write(frame)
    success, frame = camera_capture.read()

    num_frames_remaining -= 1
camera_capture.release()
