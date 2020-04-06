import numpy as np
import cv2

"""生成视差图
使用StereoSGBM
书中4.5"""
def update(val=0):
    stereo.set_block_size(cv2.get_trackbar_pos('window_size', 'disparity'))
    stereo