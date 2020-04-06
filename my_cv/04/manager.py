import cv2
import numpy as np
import time


class CaptureManager():
    def __init__(self, capture, preview_window_manager=None, should_mirror_preview=False):
        self.preview_window_manager = preview_window_manager
        self.should_mirror_preview = should_mirror_preview
        self._capture = capture
        self._channel = 0
        # 是否进入frame的标志
        self._entered_frame = False

        self._frame = None
        self._image_file_name = None
        self._video_file_name = None
        self._video_encoding = None
        self._video_writer = None
        self._start_time = None
        self._frames_elapsed = int(0)
        self._fps_estimate = None

    @property
    def channel(self):
        return self._channel

    @channel.setter
    def channel(self, value):
        if self._channel != value:
            self._channel = value
            self._frame = None

    @property
    def frame(self):
        if self._entered_frame and self._frame is None:
            _, self._frame = self._capture.retrieve()
        return self._frame

    @property
    def is_writing_image(self):
        return self._image_file_name is not None

    @property
    def is_writing_video(self):
        return self._video_file_name is not None

    def enter_frame(self):
        """如果有，获得下一个frame"""
        # 确保以前的frame已经关闭
        assert not self._entered_frame, \
            '前一个enter_frame 还没有关闭'
        if self._capture is not None:
            self._entered_frame = self._capture.grab()

    def exit_frame(self):
        """显示在窗口，写到文件，释放frame"""
        if self.frame is None:
            self._entered_frame = False
            return

        # 更新FPS估计值和相关变量
        if self._frames_elapsed == 0:
            self._start_time = time.time()
        else:
            time_elapsed = time.time() - self._start_time
            self._fps_estimate = self._frames_elapsed / time_elapsed
        self._frames_elapsed += 1
        # 如果有窗口，在窗口上显示内容
        if self.preview_window_manager is not None:
            if self.should_mirror_preview:
                mirrored_frame = np.fliplr(self._frame).copy()
                self.preview_window_manager.show(mirrored_frame)
            else:
                self.preview_window_manager.show(self._frame)
        # 如果有，就写入图片文件
        if self.is_writing_image:
            cv2.imwrite(self._image_file_name, self._frame)
            self._image_file_name = None
        # 如果有，就写入视频文件
        self._write_video_frame()
        # 释放 frame
        self._frame = None
        self._entered_frame = False

    def write_image(self, file_name):
        self._image_file_name = file_name

    def start_writing_video(self, file_name, encoding=cv2.VideoWriter_fourcc('I', '4', '2', '0')):
        self._video_file_name = file_name
        self._video_encoding = encoding

    def stop_writing_video(self):
        self._video_file_name = None
        self._video_encoding = None
        self._video_writer = None

    def _write_video_frame(self):
        if not self.is_writing_video:
            return

        if self._video_writer is None:
            fps = self._capture.get(cv2.CAP_PROP_FPS)
            if fps == 0.0:
                if self._frames_elapsed < 20:
                    return
                else:
                    fps = self._fps_estimate
            size = (int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

            self._video_writer = cv2.VideoWriter(self._video_file_name, self._video_encoding,
                                                 fps, size)
        self._video_writer.write(self._frame)


class WindowManager(object):
    def __init__(self, window_name, keypress_callback=None):
        self.keypress_callback = keypress_callback
        self._window_name = window_name
        self._is_window_created = False

    @property
    def is_window_created(self):
        return self._is_window_created

    def create_window(self):
        cv2.namedWindow(self._window_name)
        self._is_window_created = True

    def show(self, frame):
        cv2.imshow(self._window_name, frame)

    def destroy_window(self):
        cv2.destroyWindow(self._window_name)
        self._is_window_created = False

    def process_events(self):
        keycode = cv2.waitKey(1)
        if self.keypress_callback is not None and keycode != -1:
            # 除去非低位的non-ASCII信息
            keycode &= 0xFF
            self.keypress_callback(keycode)
