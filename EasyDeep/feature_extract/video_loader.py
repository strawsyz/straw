#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/2/9 17:07
# @Author  : strawsyz
# @File    : video_loader.py
# @desc:
import os
import logging
import cv2
import imutils
import numpy as np
from tqdm import tqdm
import moviepy.editor


def getDuration(video_path):
    """Get the duration (in seconds) for a video.

    Keyword arguments:
    video_path -- the path of the video
    """
    return moviepy.editor.VideoFileClip(video_path).duration


class FrameCV():
    def __init__(self, video_path, FPS=2, transform=None, start=None, duration=None):
        """Create a list of frame from a video using OpenCV.

        Keyword arguments:
        video_path -- the path of the video
        FPS -- the desired FPS for the frames (default:2)
        transform -- the desired transformation for the frames (default:2)
        start -- the desired starting time for the list of frames (default:None)
        duration -- the desired duration time for the list of frames (default:None)
        """

        self.FPS = FPS
        self.transform = transform
        self.start = start
        self.duration = duration

        # read video
        vidcap = cv2.VideoCapture(video_path)
        # read FPS
        self.fps_video = vidcap.get(cv2.CAP_PROP_FPS)
        # read duration
        self.time_second = getDuration(video_path)

        # loop until the number of frame is consistent with the expected number of frame,
        # given the duration and the FPS
        good_number_of_frames = False
        while not good_number_of_frames:

            # read video
            vidcap = cv2.VideoCapture(video_path)

            # get number of frames
            self.numframe = int(self.time_second * self.fps_video)

            # frame drop ratio
            drop_extra_frames = self.fps_video / self.FPS

            # init list of frames
            self.frames = []

            # TQDM progress bar
            pbar = tqdm(range(self.numframe), desc='Grabbing Video Frames', unit='frame')
            i_frame = 0
            ret, frame = vidcap.read()

            # loop until no frame anymore
            while ret:
                # update TQDM
                pbar.update(1)
                i_frame += 1

                # skip until starting time
                # if self.start is not None:
                #     if i_frame < self.fps_video * self.start:
                #         ret, frame = vidcap.read()
                #         continue

                # skip after duration time
                # if self.duration is not None:
                #     if i_frame > self.fps_video * (self.start + self.duration):
                #         ret, frame = vidcap.read()
                #         continue

                if (i_frame % drop_extra_frames < 1):

                    # crop keep the central square of the frame
                    if self.transform == "resize256crop224":
                        frame = imutils.resize(frame, height=256)  # keep aspect ratio
                        # number of pixel to remove per side
                        off_h = int((frame.shape[0] - 224) / 2)
                        off_w = int((frame.shape[1] - 224) / 2)
                        frame = frame[off_h:-off_h,
                                off_w:-off_w, :]  # remove pixel at each side

                    # crop remove the side of the frame
                    elif self.transform == "crop":
                        frame = imutils.resize(frame, height=224)  # keep aspect ratio
                        # number of pixel to remove per side
                        off_side = int((frame.shape[1] - 224) / 2)
                        frame = frame[:, off_side:-
                        off_side, :]  # remove them

                    # resize change the aspect ratio
                    elif self.transform == "resize":
                        # lose aspect ratio
                        frame = cv2.resize(frame, (224, 224),
                                           interpolation=cv2.INTER_CUBIC)

                    # append the frame to the list
                    self.frames.append(frame)

                # read next frame
                ret, frame = vidcap.read()

            # check if the expected number of frames were read
            if self.numframe - (i_frame + 1) <= 1:
                logging.debug("Video read properly")
                # print("Video read properly")
                good_number_of_frames = True
            else:
                logging.debug("Video NOT read properly, adjusting fps and read again")
                print("Video NOT read properly, adjusting fps and read again")
                self.fps_video = (i_frame + 1) / self.time_second

        # convert frame from list to numpy array
        self.frames = np.array(self.frames)

    def __len__(self):
        """Return number of frames."""
        return len(self.frames)

    def __iter__(self, index):
        """Return frame at given index."""
        return self.frames[index]
