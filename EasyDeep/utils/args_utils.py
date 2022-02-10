#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/2/7 14:46
# @Author  : strawsyz
# @File    : args_utils.py
# @desc:
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

parser = ArgumentParser(description='transformer for soccernet', formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument('--GPU', required=False, type=str, default=GPU, help='ID of the GPU to use')
parser.add_argument("--split_data", required=False, type=int, default=1,
                    help='split_data')
parser.add_argument("--model_names_in_type1", nargs='+', required=False, type=str,
                    default=["MyOptimizeTransformer6", "MyOptimizeTransformer7", "MyOptimizeTransformer8",
                             "MyOptimizeTransformer61"],
                    help='spot_model_path')
parser.add_argument("--test_4_highlights", required=False, type=bool, default=False,
                    help='test_4_highlights')
parser.add_argument("--loss_weight", required=False, type=float, default=False,
                    help='loss_weight')

args = parser.parse_args()


