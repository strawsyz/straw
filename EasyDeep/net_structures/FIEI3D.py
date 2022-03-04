#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/3/3 0:21
# @Author  : strawsyz
# @File    : FIEI3D.py
# @desc:
import torch

from net_structures.old_models import get_clones, SimpleEncoderLayerTrue, Norm, FIE2
from pytorch_i3d import InceptionI3d
from torch import nn


class SimpleEncoderNoPETrue(nn.Module):
    def __init__(self, vocab_size, embedding_dim, N, heads, importance_threshold=0):
        super().__init__()
        self.N = N
        self.layers = get_clones(SimpleEncoderLayerTrue(embedding_dim, heads, importance_threshold), N)
        self.norm = Norm(embedding_dim)

    def forward(self, src, mask=None):
        # Attn = []
        x = src
        for i in range(self.N):
            x, attn = self.layers[i](x, mask)
            Attn = attn
        return self.norm(x), Attn


def get_i3d_model():
    import platform
    if platform.system() == "Linux":
        model_path = r"/workspace/datasets/rgb_imagenet.pt"
    else:
        model_path = r"C:\(lab\OtherProjects\pytorch-i3d-master\models\rgb_imagenet.pt"

    i3d = InceptionI3d(400, in_channels=3)
    # i3d.replace_logits(157)
    i3d.load_state_dict(torch.load(model_path))
    i3d.replace_logits(101)

    # i3d.cuda()
    return i3d


class FIEI3DNet(nn.Module):

    def __init__(self, vocab_size=2048, embedding_dim=256, N=1, heads=1, max_seq_len=200):
        super().__init__()
        # if vocab_size != embedding_dim:
        self.linear1 = nn.Linear(vocab_size, embedding_dim)
        self.similar_filter_before = SimpleEncoderNoPETrue(embedding_dim, embedding_dim, N, heads,
                                                           importance_threshold=0)
        self.similar_filter_after = SimpleEncoderNoPETrue(embedding_dim, embedding_dim, N, heads,
                                                          importance_threshold=0)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        self.backbone = get_i3d_model()

    def forward(self, data, features):
        batch_size, num_frame, num_feature = features.shape
        # src = src[torch.randperm(src.size(0))]

        src = self.linear1(features)
        nb_frames_half = int(num_frame / 2)
        src_before = src[:, :nb_frames_half, :]
        src_after = src[:, nb_frames_half:, :]
        # output_before, _ = self.similar_filter_before(src_before, None)
        # output_after, _ = self.similar_filter_after(src_after, None)
        _, output_before = self.similar_filter_before(src_before, None)
        _, output_after = self.similar_filter_after(src_after, None)
        attn = torch.cat((output_before.mean(dim=1), output_after.mean(dim=1)), dim=1)
        data = data.permute(1, 3, 4, 0, 2) * attn
        output = self.backbone(data.permute(3, 0, 4, 1, 2))
        return output


class FIEI3DNet2(nn.Module):

    def __init__(self, vocab_size=2048, embedding_dim=256, N=1, heads=1, max_seq_len=200, num_class=101):
        super().__init__()

        self.fie = FIE2(vocab_size, embedding_dim, N, heads, num_class, max_seq_len=200)
        self.backbone = get_i3d_model()

    def forward(self, data, features):
        output_fie = self.fie(features)
        output = self.backbone(data.permute(3, 0, 4, 1, 2))
        output = output.squeeze(dim=2)
        return output + output_fie

# class FIEI3DNet2(nn.Module):
#
#     def __init__(self, vocab_size=2048, embedding_dim=256, N=1, heads=1, max_seq_len=200, num_class=101):
#         super().__init__()
#
#         # self.linear1 = nn.Linear(vocab_size, embedding_dim)
#         # self.similar_filter_before = SimpleEncoderNoPETrue(embedding_dim, embedding_dim, N, heads,
#         #                                                    importance_threshold=0)
#         # self.similar_filter_after = SimpleEncoderNoPETrue(embedding_dim, embedding_dim, N, heads,
#         #                                                   importance_threshold=0)
#         # self.sigmoid = nn.Sigmoid()
#         # self.out = nn.Linear(embedding_dim, num_class)
#
#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)
#         self.fie = FIE2(vocab_size, embedding_dim, N, heads, num_class, max_seq_len=200)
#         self.backbone = get_i3d_model()
#
#     def forward(self, data, features):
#         batch_size, num_frame, num_feature = features.shape
#
#         src = self.linear1(features)
#         nb_frames_half = int(num_frame / 2)
#         src_before = src[:, :nb_frames_half, :]
#         src_after = src[:, nb_frames_half:, :]
#         output_before, _ = self.similar_filter_before(src_before, None)
#         output_after, _ = self.similar_filter_after(src_after, None)
#         # _, output_before = self.similar_filter_before(src_before, None)
#         # _, output_after = self.similar_filter_after(src_after, None)
#         attn = torch.cat((output_before.mean(dim=1), output_after.mean(dim=1)), dim=1)
#         data = data.permute(1, 3, 4, 0, 2) * attn
#         batch_size, num_frame, num_feature = src.shape
#         # src = src[torch.randperm(src.size(0))]
#
#         src = self.linear1(src)
#         nb_frames_half = int(num_frame / 2)
#         src_before = src[:, :nb_frames_half, :]
#         src_after = src[:, nb_frames_half:, :]
#         output_before, _ = self.similar_filter_before(src_before, None)
#         output_after, _ = self.similar_filter_after(src_after, None)
#         output = torch.cat((output_before, output_after), dim=1)
#
#         output = output.mean(dim=1)
#         output = self.sigmoid(output)
#         output_fie = self.out(output)
#
#         output = self.backbone(data.permute(3, 0, 4, 1, 2))
#         output.squeeze(dim=2)
#         return output + output_fie
