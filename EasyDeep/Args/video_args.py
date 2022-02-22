#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/2/10 14:49
# @Author  : strawsyz
# @File    : video_args.py
# @desc:

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


class VideoArgs():
    def __init__(self):
        GPU = "0"

        parser = ArgumentParser(description='transformer for soccernet', formatter_class=ArgumentDefaultsHelpFormatter)

        parser.add_argument('--GPU', required=False, type=str, default=GPU, help='ID of the GPU to use')
        parser.add_argument('--embeddings_dim', required=False, type=int, default=512, help='Embedding Size')
        parser.add_argument('--heads', required=False, type=int, default=1, help='Heads')
        parser.add_argument('--n_layers', required=False, type=int, default=1, help='Num of Layers')
        parser.add_argument('--batch_size', required=False, type=int, default=64, help='Batch Size')
        parser.add_argument('--LR', required=False, type=float, default=0.001, help='Learning Rate')
        parser.add_argument('--max_try_times', required=False, type=int, default=8, help='Max Try Times')
        parser.add_argument('--clip_length', required=False, type=int, default=18, help='Number of frames in Clip')
        parser.add_argument('--model_name', required=False, type=str, default="Transformer", help='Model Name')

        # parser.add_argument("--split_data", required=False, type=int, default=1,
        #                     help='split_data')
        # parser.add_argument("--model_names_in_type1", nargs='+', required=False, type=str,
        #                     default=["MyOptimizeTransformer6", "MyOptimizeTransformer7", "MyOptimizeTransformer8",
        #                              "MyOptimizeTransformer61"],
        #                     help='spot_model_path')
        # parser.add_argument("--test_4_highlights", required=False, type=bool, default=False,
        #                     help='test_4_highlights')
        # parser.add_argument("--loss_weight", required=False, type=float, default=False,
        #                     help='loss_weight')

        self.args = parser.parse_args()

    def get_args(self):
        return self.args


args = VideoArgs()


def get_args():
    return args.get_args()
