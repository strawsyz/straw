#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/10/8 20:47
# @Author  : strawsyz
# @File    : test_image_example.py
# @desc:
from unittest import TestCase

from examples.image_example import ImageExperiment
from utils.estimate_utils import iou_estimate


def pretrained_model_predict_test_dataset(experiment, pretrain_path):
    experiment.pretrain_path = pretrain_path.strip()
    experiment.is_pretrain = True
    result_path = experiment.test(save_predict_result=True)
    return result_path


if __name__ == '__main__':
    # gt_data_path = r"/home/shi/Downloads/dataset/polyp/TMP/09/test/mask"
    gt_data_path = r"/home/shi/Downloads/dataset/Kvasir-SEG/masks"

    # trained model's path
    # pretrain_path = r"/home/shi/Downloads/models/polyp/2021-03-04/ep21_20-05-02.pkl"
    # pretrain_path = r"/home/shi/Downloads/models/polyp/2021-03-13/ep64_22-04-44.pkl"  # max average calcu_iou is 0.6529785990715027
    # pretrain_path = r"/home/shi/Downloads/models/polyp/2021-03-14/ep27_00-06-44.pkl"  # 0.48169809579849243   # 0.6407430768013
    # pretrain_path = r"/home/shi/Downloads/models/polyp/2021-03-14/ep31_11-48-12.pkl"  # 0.7375684380531311
    # with edge. with 0.8 * iamge + 0.2 edge  # 0.7375684380531311
    # [INFO]<2021-03-14 11:47:59,582> { EPOCH:31	 train_loss:0.305649 }
    # [INFO]<2021-03-14 11:48:12,559> { Epoch:31	 valid_loss:0.485193 }
    # [INFO]<2021-03-14 11:48:12,560> { ========== saving experiment history ========== }
    # [INFO]<2021-03-14 11:48:12,561> { ========== saved experiment history at /home/shi/Downloads/models/polyp/history/edge_03-14_10-23-29.pth========== }
    # [INFO]<2021-03-14 11:48:12,562> { save this model for ['train_loss', 'valid_loss'] is better }
    # [INFO]<2021-03-14 11:48:12,562> { ==============saving model data=============== }
    # [INFO]<2021-03-14 11:48:12,819> { ==============saved at /home/shi/Downloads/models/polyp/2021-03-14/ep31_11-48-12.pkl=============== }
    # [INFO]<2021-03-14 11:48:12,819> { 0.30564871430397034 is best score of train_loss, saved at /home/shi/Downloads/models/polyp/2021-03-14/ep31_11-48-12.pkl }
    # [INFO]<2021-03-14 11:48:12,819> { 0.48519277572631836 is best score of valid_loss, saved at /home/shi/Downloads/models/polyp/2021-03-14/ep31_11-48-12.pkl }
    # [INFO]<2021-03-14 11:48:12,819> { use 158 seconds in the epoch }
    # with edge. with 0.7 * iamge + 0.3 edge
    # pretrain_path = r"/home/shi/Downloads/models/polyp/2021-03-14/ep21_14-49-13.pkl"   # 0.6499894261360168
    # [INFO]<2021-03-14 14:49:00,122> { EPOCH:21	 train_loss:0.364955 }
    # [INFO]<2021-03-14 14:49:13,157> { Epoch:21	 valid_loss:0.497743 }
    # [INFO]<2021-03-14 14:49:13,157> { ========== saving experiment history ========== }
    # [INFO]<2021-03-14 14:49:13,158> { ========== saved experiment history at /home/shi/Downloads/models/polyp/history/edge_03-14_13-50-57.pth========== }
    # [INFO]<2021-03-14 14:49:13,159> { save this model for ['train_loss', 'valid_loss'] is better }
    # [INFO]<2021-03-14 14:49:13,159> { ==============saving model data=============== }
    # [INFO]<2021-03-14 14:49:13,402> { ==============saved at /home/shi/Downloads/models/polyp/2021-03-14/ep21_14-49-13.pkl=============== }
    # [INFO]<2021-03-14 14:49:13,403> { 0.36495473980903625 is best score of train_loss, saved at /home/shi/Downloads/models/polyp/2021-03-14/ep21_14-49-13.pkl }
    # [INFO]<2021-03-14 14:49:13,403> { 0.49774327874183655 is best score of valid_loss, saved at /home/shi/Downloads/models/polyp/2021-03-14/ep21_14-49-13.pkl }
    # [INFO]<2021-03-14 14:49:13,403> { use 158 seconds in the epoch }
    # with edge. 0.9 image + 0.1 edge 0.476624    # 0.75467503
    # pretrain_path = r"/home/shi/Downloads/models/polyp/2021-03-14/ep24_17-00-25.pkl"
    #  { EPOCH:24	 train_loss:0.319781 }
    # [INFO]<2021-03-14 17:00:25,563> { Epoch:24	 valid_loss:0.476624 }
    # [INFO]<2021-03-14 17:00:25,563> { ========== saving experiment history ========== }
    # [INFO]<2021-03-14 17:00:25,564> { ========== saved experiment history at /home/shi/Downloads/models/polyp/history/edge_03-14_15-54-06.pth========== }
    # [INFO]<2021-03-14 17:00:25,565> { save this model for ['train_loss', 'valid_loss'] is better }
    # [INFO]<2021-03-14 17:00:25,565> { ==============saving model data=============== }
    # [INFO]<2021-03-14 17:00:25,827> { ==============saved at /home/shi/Downloads/models/polyp/2021-03-14/ep24_17-00-25.pkl=============== }
    # [INFO]<2021-03-14 17:00:25,827> { 0.31978121399879456 is best score of train_loss, saved at /home/shi/Downloads/models/polyp/2021-03-14/ep24_17-00-25.pkl }
    # [INFO]<2021-03-14 17:00:25,827> { 0.4766244888305664 is best score of valid_loss, saved at /home/shi/Downloads/models/polyp/2021-03-14/ep24_17-00-25.pkl }
    # [INFO]<2021-03-14 17:00:25,827> { use 158 seconds in the epoch }
    # no edge /home/shi/Downloads/models/polyp/2021-03-14/ep21_18-33-46.pkl   0.7903040647506714
    # pretrain_path = r"/home/shi/Downloads/models/polyp/2021-03-14/ep21_18-33-46.pkl"
    # { EPOCH:21	 train_loss:0.327205 }
    # [INFO]<2021-03-14 18:33:46,348> { Epoch:21	 valid_loss:0.460770 }
    # [INFO]<2021-03-14 18:33:46,348> { ========== saving experiment history ========== }
    # [INFO]<2021-03-14 18:33:46,349> { ========== saved experiment history at /home/shi/Downloads/models/polyp/history/None_03-14_17-35-42.pth========== }
    # [INFO]<2021-03-14 18:33:46,350> { save this model for ['train_loss', 'valid_loss'] is better }
    # [INFO]<2021-03-14 18:33:46,350> { ==============saving model data=============== }
    # [INFO]<2021-03-14 18:33:46,609> { ==============saved at /home/shi/Downloads/models/polyp/2021-03-14/ep21_18-33-46.pkl=============== }
    # [INFO]<2021-03-14 18:33:46,610> { 0.32720497250556946 is best score of train_loss, saved at /home/shi/Downloads/models/polyp/2021-03-14/ep21_18-33-46.pkl }
    # [INFO]<2021-03-14 18:33:46,610> { 0.4607701301574707 is best score of valid_loss, saved at /home/shi/Downloads/models/polyp/2021-03-14/ep21_18-33-46.pkl }
    # [INFO]<2021-03-14 18:33:46,610> { use 158 seconds in the epoch }
    # edge 0.1
    pretrain_path = r"/home/shi/Downloads/models/polyp/2021-03-15/ep28_10-12-27.pkl"
    # { EPOCH:28	 train_loss:0.310630 }
    # [INFO]<2021-03-15 10:12:27,227> { Epoch:28	 valid_loss:0.477860 }
    # [INFO]<2021-03-15 10:12:27,227> { ========== saving experiment history ========== }
    # [INFO]<2021-03-15 10:12:27,229> { ========== saved experiment history at /home/shi/Downloads/models/polyp/history/edge_03-15_08-55-03.pth========== }
    # [INFO]<2021-03-15 10:12:27,229> { save this model for ['train_loss', 'valid_loss'] is better }
    # [INFO]<2021-03-15 10:12:27,229> { ==============saving model data=============== }
    # [INFO]<2021-03-15 10:12:27,471> { ==============saved at /home/shi/Downloads/models/polyp/2021-03-15/ep28_10-12-27.pkl=============== }
    # [INFO]<2021-03-15 10:12:27,471> { 0.31063029170036316 is best score of train_loss, saved at /home/shi/Downloads/models/polyp/2021-03-15/ep28_10-12-27.pkl }
    # [INFO]<2021-03-15 10:12:27,472> { 0.47786012291908264 is best score of valid_loss, saved at /home/shi/Downloads/models/polyp/2021-03-15/ep28_10-12-27.pkl }
    # [INFO]<2021-03-15 10:12:27,472> { use 159 seconds in the epoch }
    # pretrain_path = r"/home/shi/Downloads/models/ep64_22-04-44.pkl"
    # pretrain_path = r"/home/shi/Downloads/models/ep21_20-05-02.pkl"

    # pretrain_path = r"/home/shi/Downloads/models/ep31_11-48-12-with_edge_best.pkl"
    # pretrain_path = r"/home/shi/Downloads/models/polyp/2021-03-16/ep0_09-35-43.pkl"  # loss is 0.89, worth try (m ,m)

    pretrain_path = r"/home/shi/Downloads/models/polyp/2021-03-17/ep2_11-00-33.pkl"  # 0.59
    pretrain_path = r"/home/shi/Downloads/models/polyp/2021-03-17/ep4_11-06-26.pkl"
    pretrain_path = r"/home/shi/Downloads/models/polyp/2021-03-17/ep12_11-29-57.pkl"  # 0.7375684380531311
    pretrain_path = r"/home/shi/Downloads/models/polyp/2021-03-24/ep189_17-30-39.pkl"
    pretrain_path = r"/home/shi/Downloads/models/polyp/2021-03-24/ep278_20-56-17.pkl"
    pretrain_path = r"/home/shi/Downloads/models/polyp/2021-03-25/ep176_01-58-49.pkl"
    pretrain_path = r"/home/shi/Downloads/models/polyp/2021-03-25/ep262_00-28-42.pkl"
    pretrain_path = r"/home/shi/Downloads/models/polyp/2021-03-25/ep206_13-21-41.pkl"
    experiment = ImageExperiment()
    image_path = r"/home/shi/Downloads/dataset/Kvasir-SEG/images/cju0qkwl35piu0993l0dewei2.jpg"
    mask_path = r"/home/shi/Downloads/dataset/Kvasir-SEG/masks/cju0qkwl35piu0993l0dewei2.jpg"
    experiment.pretrain_path = pretrain_path.strip()
    experiment.is_pretrain = True
    # experiment.test_file(image_path, mask_path)

    # experiment = ImageExperiment(tag="edge")

    # experiment.history_save_path = r""
    # experiment.estimate(save_path="tmp.png", use_log10=True)

    # predict_data_path = r'/home/shi/Download\models\polyp/result/trained_resnet34'
    # predict_data_path = r'/home/shi/Download\models\polyp/result/resnet50-unpretrained-01'

    # result_path = r"/home/shi/Download\models\polyp/result/2021-03-04/"  # max average calcu_iou is 0.7256231904029846
    result_path = pretrained_model_predict_test_dataset(experiment, pretrain_path)
    iou_estimate(gt_data_path, result_path)
    # iou_estimate(gt_data_path, result_path, thresholds=range(1, 2))
# 0.49413299560546875
# 0.49469056725502014


# [INFO]<2021-03-14 00:06:44,748> { Epoch:27	 valid_loss:0.481698 }
# [INFO]<2021-03-14 00:06:44,748> { ========== saving experiment history ========== }
# [INFO]<2021-03-14 00:06:44,750> { ========== saved experiment history at /home/shi/Downloads/models/polyp/history/edge_03-13_22-52-27.pth========== }
# [INFO]<2021-03-14 00:06:44,750> { save this model for ['valid_loss'] is better }
# [INFO]<2021-03-14 00:06:44,750> { ==============saving model data=============== }
# [INFO]<2021-03-14 00:06:45,010> { ==============saved at /home/shi/Downloads/models/polyp/2021-03-14/ep27_00-06-44.pkl=============== }
# [INFO]<2021-03-14 00:06:45,010> { 0.3155905306339264 is best score of train_loss, saved at /home/shi/Downloads/models/polyp/2021-03-14/ep27_00-06-44.pkl }
# [INFO]<2021-03-14 00:06:45,010> { 0.48169809579849243 is best score of valid_loss, saved at /home/shi/Downloads/models/polyp/2021-03-14/ep27_00-06-44.pkl }
# [INFO]<2021-03-14 00:06:45,010> { use 159 seconds in the epoch }

# max average calcu_iou is 0.5881792902946472    79
# resnet 50
# { EPOCH:9	 train_loss:0.372698 }
# [INFO]<2021-03-15 16:25:15,564> { Epoch:9	 valid_loss:0.470246 }
# [INFO]<2021-03-15 16:25:15,564> { ========== saving experiment history ========== }
# [INFO]<2021-03-15 16:25:15,565> { ========== saved experiment history at /home/shi/Downloads/models/polyp/history/None_03-15_15-45-11.pth========== }
# [INFO]<2021-03-15 16:25:15,565> { save this model for ['train_loss', 'valid_loss'] is better }
# [INFO]<2021-03-15 16:25:15,566> { ==============saving model data=============== }
# [INFO]<2021-03-15 16:25:16,644> { ==============saved at /home/shi/Downloads/models/polyp/2021-03-15/ep9_16-25-15.pkl=============== }
# [INFO]<2021-03-15 16:25:16,645> { 0.3726975619792938 is best score of train_loss, saved at /home/shi/Downloads/models/polyp/2021-03-15/ep9_16-25-15.pkl }
# [INFO]<2021-03-15 16:25:16,645> { 0.4702455401420593 is best score of valid_loss, saved at /home/shi/Downloads/models/polyp/2021-03-15/ep9_16-25-15.pkl }
# [INFO]<2021-03-15 16:25:16,645> { use 702 seconds in the epoch }
