import cv2
from manager import WindowManager, CaptureManager
import filters

"""摄像头视频录制功能
点击 空格键 ，截图
点击 tab键 开始、停止录屏
点击 退出键 退出程序
相比第二章中的cameo，增加了对图像的处理"""


class Cameo(object):
    def __init__(self):
        self._window_manager = WindowManager('Cameo', self.on_key_press)
        self._capture_manager = CaptureManager(cv2.VideoCapture(0), self._window_manager, True)
        self._curve_filter = filters.BGRPortraCurveFilter()

    def run(self):
        self._window_manager.create_window()
        while self._window_manager.is_window_created:
            self._capture_manager.enter_frame()
            frame = self._capture_manager.frame
            # 描绘边缘
            filters.stroke_edges(frame, frame)
            # 模拟肖像色彩
            self._curve_filter.apply(frame, frame)
            self._capture_manager.exit_frame()
            self._window_manager.process_events()

    def on_key_press(self, keycode):
        # 如果点击了空格键
        if keycode == 32:
            self._capture_manager.write_image('screenshot.png')
        # 如果点击了tab键
        elif keycode == 9:
            # 切换是否开始录制视频
            if not self._capture_manager.is_writing_video:
                self._capture_manager.start_writing_video('screencast.avi')
            else:
                self._capture_manager.stop_writing_video()
        # 如果点击了退出键
        elif keycode == 27:
            self._window_manager.destroy_window()


if __name__ == '__main__':
    Cameo().run()
