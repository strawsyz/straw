#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/2/3 14:15
# @Author  : strawsyz
# @File    : feature_util.py
# @desc:


import os
# from keras.applications.resnet import preprocess_input
import numpy as np
import imutils  # pip install imutils
from tqdm import tqdm

import logging
import cv2
import moviepy.editor

# TODO 环境问题需要修改，其他基本没问题
from pytorch_i3d import InceptionI3d


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
        # given the duratio nand the FPS
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
                if self.start is not None:
                    if i_frame < self.fps_video * self.start:
                        ret, frame = vidcap.read()
                        continue

                # skip after duration time
                if self.duration is not None:
                    if i_frame > self.fps_video * (self.start + self.duration):
                        ret, frame = vidcap.read()
                        continue

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
                good_number_of_frames = True
            else:
                logging.debug("Video NOT read properly, adjusting fps and read again")
                self.fps_video = (i_frame + 1) / self.time_second

        # convert frame from list to numpy array
        self.frames = np.array(self.frames)

    def __len__(self):
        """Return number of frames."""
        return len(self.frames)

    def __iter__(self, index):
        """Return frame at given index."""
        return self.frames[index]


def extract_features(video_path, feature_path, model, start=None, duration=None, overwrite=False, FPS=2,
                     transform="crop"):
    print("extract video", video_path, "from", start, duration)
    # feature_path = video_path.replace(
    #     ".mkv", f"_{self.feature}_{self.back_end}.npy")
    # feature_path = video_path[:-4] + f"_{self.feature}_{self.back_end}.npy"

    if os.path.exists(feature_path) and not overwrite:
        return

    # if self.grabber == "skvideo":
    #     videoLoader = Frame(video_path, FPS=self.FPS, transform=self.transform, start=start, duration=duration)
    # elif self.grabber == "opencv":
    videoLoader = FrameCV(video_path, FPS=FPS, transform=transform, start=start,
                          duration=duration)

    # create numpy aray (nb_frames x 224 x 224 x 3)
    # frames = np.array(videoLoader.frames)
    # if self.preprocess:
    # frames = preprocess_input(videoLoader.frames)
    frames = videoLoader.frames

    if duration is None:
        duration = videoLoader.time_second
        # time_second = duration
    print("frames", frames.shape, "fps=", frames.shape[0] / duration)
    # from torch.nn import Variable
    from torch.autograd import Variable

    frames = Variable(torch.from_numpy(frames))
    frames = frames.permute([3, 0, 1, 2])
    frames = frames.unsqueeze(0)
    frames = frames.cuda()
    out = model(frames)
    out = out.squeeze(0)
    features = out.cpu().detach().numpy()

    # predict the featrues from the frames (adjust batch size for smalled GPU)
    # features = model.predict(frames, batch_size=64, verbose=1)
    # print("features", features.shape, "fps=", features.shape[0] / duration)

    num_frames = features.shape[0]

    # save the featrue in .npy format
    os.makedirs(os.path.dirname(feature_path), exist_ok=True)
    np.save(feature_path, features)
    print(f"Save  features at {feature_path}")
    return num_frames


def extract_features(video_path, feature_path, model, start=None, duration=None, overwrite=False, FPS=2,
                     transform="crop"):
    print("extract video", video_path, "from", start, duration)
    # feature_path = video_path.replace(
    #     ".mkv", f"_{self.feature}_{self.back_end}.npy")
    # feature_path = video_path[:-4] + f"_{self.feature}_{self.back_end}.npy"

    if os.path.exists(feature_path) and not overwrite:
        return

    # if self.grabber == "skvideo":
    #     videoLoader = Frame(video_path, FPS=self.FPS, transform=self.transform, start=start, duration=duration)
    # elif self.grabber == "opencv":
    videoLoader = FrameCV(video_path, FPS=FPS, transform=transform, start=start,
                          duration=duration)

    # create numpy aray (nb_frames x 224 x 224 x 3)
    # frames = np.array(videoLoader.frames)
    # if self.preprocess:
    # frames = preprocess_input(videoLoader.frames)
    frames = videoLoader.frames

    if duration is None:
        duration = videoLoader.time_second
        # time_second = duration
    print("frames", frames.shape, "fps=", frames.shape[0] / duration)
    # from torch.nn import Variable
    from torch.autograd import Variable

    frames = Variable(torch.from_numpy(frames))
    frames = frames.permute([3, 0, 1, 2])
    frames = frames.unsqueeze(0)
    frames = frames.cuda()
    out = model(frames)
    out = out.squeeze(0)
    features = out.cpu().detach().numpy()

    # predict the featrues from the frames (adjust batch size for smalled GPU)
    # features = model.predict(frames, batch_size=64, verbose=1)
    # print("features", features.shape, "fps=", features.shape[0] / duration)

    num_frames = features.shape[0]

    # save the featrue in .npy format
    os.makedirs(os.path.dirname(feature_path), exist_ok=True)
    np.save(feature_path, features)
    print(f"Save  features at {feature_path}")
    return num_frames


def analyze(self, data: list):
    print(f"min : {min(data)}")
    print(f"max : {max(data)}")
    print(f"mean : {np.mean(data)}")
    print(f"middle : {np.middle(data)}")


def get_i3d_model(model_path=r"C:\(lab\OtherProjects\pytorch-i3d-master\models\rgb_imagenet.pt"):
    i3d = InceptionI3d(400, in_channels=3)
    i3d.replace_logits(157)
    i3d.load_state_dict(torch.load(model_path))
    i3d.cuda()
    return i3d


if __name__ == "__main__":
    from timesformer.models.vit import TimeSformer
    import torch

    # pretrain_path = r"C:\(lab\PretrainedModel\TimeSFormer\TimeSformer_divST_96x4_224_K400.pyth"
    #
    # device = torch.device("cuda:0")
    #
    # model = TimeSformer(img_size=224, num_classes=400, num_frames=8, attention_type='divided_space_time',
    #                     pretrained_model=pretrain_path)
    #
    # model = model.eval().to(device)

    # video_path = r"C:\(lab\datasets\UCF101\train\ApplyEyeMakeup\v_ApplyEyeMakeup_g08_c01.avi"
    # output_path = r"C:\(lab\datasets\UCF101\timesformer\1.npy"
    # output_path = r"C:\(lab\datasets\UCF101\i3d\1.npy"
    #
    # feature_path = r"C:\(lab\datasets\UCF101\features\resize-resnet"
    # video_path = r"C:\(lab\datasets\UCF101\val"

    video_path =r"C:\(lab\datasets\UCF101\train\ApplyEyeMakeup\v_ApplyEyeMakeup_g08_c04.avi"
    output_path = r"1.npy"

    model = get_i3d_model()
    num_frames = extract_features(video_path=video_path, feature_path=output_path, model=model)
    # print(num_frames)
    # transform = "resize"
    # # all_num_frames = []
    # for label in os.listdir(video_path):
    #     for filename in os.listdir(os.path.join(video_path, label)):
    #         video_sample_path = os.path.join(video_path, label, filename)
    #         feature_folder_path = os.path.join(feature_path, label)
    #         feature_sample_path = os.path.join(feature_folder_path, filename.split(".")[0])
    #         # make_directory(feature_path)
    #         print(f"video path : {video_sample_path}")
    #         print(f"feature path : {feature_sample_path}")
    #         num_frames = extract_features(video_path=video_sample_path, feature_path=feature_sample_path,
    #                                       model=model, transform=transform)
            # all_num_frames.append(num_frames)
    #
    # analyze(all_num_frames)
