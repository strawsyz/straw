# -*- coding: utf-8 -*-
# File  : video_utils.py
# Author: strawsyz
# Date  : 2023/7/21
import os.path

import cv2

def extract_frames_from_video(video_path, image_save_root_path,num_key_frames):
    cap = cv2.VideoCapture(video_path)
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 生成目标帧的索引
    frame_indexes = [i for i in range(0, total_frame-1, int(total_frame/num_key_frames))]
    # frame_indexes.append(total_frame-1)
    assert len(frame_indexes) == num_key_frames
    COUNT = 0
    while True:
        success, frame = cap.read()
        if success:
            if COUNT in frame_indexes:  # 如果是需要保存的帧
                # save the frame
                image_filepath = os.path.join(image_save_root_path, f"{COUNT}.jpg")
                cv2.imwrite(image_filepath, frame)
            COUNT += 1
        else:
            if COUNT > frame_indexes[-1]:
                break
            else:
                print("Cap read result", success)
                print("total_frame", total_frame)
                print("COUNT", COUNT)
                raise RuntimeError("can not extract the frame")


if __name__ == '__main__':
    # video_path = r"C:\(lab\datasets\UCF-Crime\Anomaly\Arson\Arson022_x264.mp4"
    # image_save_root_path= r"C:\(lab\datasets\UCF-Crime\tmp"
    # video_path = r"C:\(lab\datasets\UCF-Crime\tmp\Robbery128_x264-4.avi"
    # image_save_root_path = r"C:\(lab\datasets\UCF-Crime\tmp\1"

    # video_path = r"C:\(lab\datasets\UCF-Crime\tmp\Robbery144_x264-29.avi"
    # image_save_root_path = r"C:\(lab\datasets\UCF-Crime\tmp\2"

    video_path = r"C:\(lab\datasets\UCF-Crime\tmp\Robbery092_x264-15.avi"
    image_save_root_path = r"C:\(lab\datasets\UCF-Crime\tmp\3"

    num_key_frames = 5
    extract_frames_from_video(video_path, image_save_root_path,num_key_frames)
