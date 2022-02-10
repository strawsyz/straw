#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2021/09/17 17:16
# @Author  : strawsyz
# @File    : model.py
# @desc:
# from model_0917 import *
from enum import Enum

from torch import nn
from torch.nn import Transformer, TransformerEncoderLayer, LayerNorm, TransformerEncoder

import torch
import torch.nn.functional as F
import math
from scipy.spatial.distance import cosine
import copy

from torch.autograd import Variable
import numpy as np

def attention(q, k, v, d_k, mask=None, dec_mask=False, important_threshold=0):
    scores = torch.matmul(q, k.transpose(-2, -1))
    # scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    # if mask is not None:
    #     if dec_mask:
    #         mask = mask.view(mask.size(0), 1, mask.size(1), mask.size(2))
    #     else:
    #         mask = mask.unsqueeze(1)
    #     scores = scores.masked_fill(mask == 0, -1e9)
    scores = scores - important_threshold
    scores = F.softmax(scores, dim=-1)

    output = torch.matmul(scores, v)
    return output, scores


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Norm(nn.Module):
    def __init__(self, embedding_dim, eps=1e-6):
        super().__init__()

        self.size = embedding_dim

        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))

        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

class FrameNorm(nn.Module):
    def __init__(self, num_frame, eps=1e-6):
        super().__init__()

        self.size = num_frame

        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))

        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-2, keepdim=True)) \
               / (x.std(dim=-2, keepdim=True) + self.eps) + self.bias
        return norm


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, embedding_dim):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.d_k = embedding_dim // heads
        self.h = heads

        self.q_linear = nn.Linear(embedding_dim, embedding_dim)
        self.v_linear = nn.Linear(embedding_dim, embedding_dim)
        self.k_linear = nn.Linear(embedding_dim, embedding_dim)

        # self.out = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, q, k, v, mask=None, dec_mask=False):
        bs = q.size(0)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        scores, attn = attention(q, k, v, self.d_k, mask, dec_mask)

        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.embedding_dim)
        output = self.out(concat)

        return output, attn


class SimpleMultiHeadAttention(nn.Module):
    def __init__(self, heads, embedding_dim):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.d_k = embedding_dim // heads
        self.h = heads

        self.out = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, q, k, v, mask=None, dec_mask=False):
        bs = q.size(0)

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        scores, attn = attention(q, k, v, self.d_k, mask, dec_mask)

        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.embedding_dim)
        output = self.out(concat)

        return output, attn


class SimpleMultiHeadAttentionTrue(nn.Module):
    def __init__(self, heads, embedding_dim, importance_threshold=0):
        super().__init__()
        assert heads == 1
        self.embedding_dim = embedding_dim
        self.d_k = embedding_dim // heads
        self.h = heads
        self.importance_threshold = importance_threshold

        # self.out = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, q, k, v, mask=None, dec_mask=False):
        bs = q.size(0)
        #
        # k = k.view(bs, -1, self.h, self.d_k)
        # q = q.view(bs, -1, self.h, self.d_k)
        # v = v.view(bs, -1, self.h, self.d_k)

        # k = k.transpose(1, 2)
        # q = q.transpose(1, 2)
        # v = v.transpose(1, 2)

        scores, attn = attention(q, k, v, self.d_k, mask, dec_mask, self.importance_threshold)

        # concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.embedding_dim)
        # output = self.out(concat)

        return scores, attn


class FeedForward(nn.Module):
    def __init__(self, embedding_dim, d_ff=2048):
        super().__init__()

        self.linear_1 = nn.Linear(embedding_dim, d_ff)
        self.linear_2 = nn.Linear(d_ff, embedding_dim)

    def forward(self, x):
        x = F.relu(self.linear_1(x))
        x = self.linear_2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, heads):
        super().__init__()
        self.norm_1 = Norm(embedding_dim)
        self.norm_2 = Norm(embedding_dim)
        self.attn = MultiHeadAttention(heads, embedding_dim)
        self.ff = FeedForward(embedding_dim)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        attn, enc_attn = self.attn(x2, x2, x2, mask, dec_mask=False)
        x = x + attn
        x2 = self.norm_2(x)
        x = x + self.ff(x2)
        return x, enc_attn


class SimpleEncoderLayer(nn.Module):
    def __init__(self, embedding_dim, heads):
        super().__init__()
        self.norm_1 = Norm(embedding_dim)
        self.norm_2 = Norm(embedding_dim)
        self.attn = SimpleMultiHeadAttention(heads, embedding_dim)
        self.ff = FeedForward(embedding_dim)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        attn, enc_attn = self.attn(x2, x2, x2, mask, dec_mask=False)
        x = x + attn
        x2 = self.norm_2(x)
        x = x + self.ff(x2)
        return x, enc_attn


class SimpleEncoderLayerTrue(nn.Module):
    def __init__(self, embedding_dim, heads, importance_threshold=0):
        super().__init__()
        self.norm_1 = Norm(embedding_dim)
        self.norm_2 = Norm(embedding_dim)
        self.attn = SimpleMultiHeadAttentionTrue(heads, embedding_dim, importance_threshold)
        self.ff = FeedForward(embedding_dim, d_ff=2 * embedding_dim)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        attn, enc_attn = self.attn(x2, x2, x2, mask, dec_mask=False)
        x = x + attn
        x2 = self.norm_2(x)
        x = x + self.ff(x2)
        return x, enc_attn


class MostSimpleEncoderLayer(nn.Module):
    def __init__(self, embedding_dim, heads):
        super().__init__()
        self.norm_1 = Norm(embedding_dim)
        # self.norm_2 = Norm(embedding_dim)
        self.attn = SimpleMultiHeadAttentionTrue(heads, embedding_dim)
        # self.ff = FeedForward(embedding_dim, d_ff=4 * embedding_dim)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        attn, enc_attn = self.attn(x2, x2, x2, mask, dec_mask=False)
        # x = x + attn
        # x2 = self.norm_2(x)
        # x = x + x2
        return F.relu(attn), enc_attn


class MixEncoderLayer(nn.Module):
    def __init__(self, embedding_dim, heads):
        super().__init__()
        self.norm_1 = Norm(embedding_dim)
        self.norm_2 = Norm(embedding_dim)
        self.simple_attn = SimpleMultiHeadAttention(heads, embedding_dim)
        self.attn = MultiHeadAttention(heads, embedding_dim)
        self.ff = FeedForward(embedding_dim)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        simple_attn, simple_enc_attn = self.simple_attn(x2, x2, x2, mask, dec_mask=False)
        attn, enc_attn = self.attn(x2, x2, x2, mask, dec_mask=False)
        x = x + attn + simple_attn
        x2 = self.norm_2(x)
        x = x + self.ff(x2)
        return x, enc_attn


class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, N, heads):
        super().__init__()
        self.N = N
        self.pe = PositionalEncoder(embedding_dim)
        # print(vocab_size, embedding_dim)
        # self.src_word_emb = nn.Embedding(vocab_size, embedding_dim, padding_idx=None)
        # self.ln = nn.Linear(vocab_size, embedding_dim)
        self.layers = get_clones(EncoderLayer(embedding_dim, heads), N)
        self.norm = Norm(embedding_dim)
        if vocab_size == embedding_dim:
            self.no_embedding = True
        else:
            self.no_embedding = False
            self.src_word_emb = nn.Linear(vocab_size, embedding_dim)

    def forward(self, src, mask):
        if not self.no_embedding:
            src = self.src_word_emb(src)
        x = self.pe(src)
        Attn = []
        for i in range(self.N):
            x, attn = self.layers[i](x, mask)
            Attn.append(attn)
        return self.norm(x), Attn


class SimpleEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, N, heads, max_seq_len=200, pe_type=None):
        super().__init__()
        self.N = N
        self.pe = PositionalEncoder(embedding_dim, max_seq_len=max_seq_len, type=pe_type)
        self.layers = get_clones(SimpleEncoderLayer(embedding_dim, heads), N)
        self.norm = Norm(embedding_dim)
        if vocab_size == embedding_dim:
            self.no_embedding = True
        else:
            self.no_embedding = False
            self.src_word_emb = nn.Linear(vocab_size, embedding_dim)

    def forward(self, src, mask=None):
        if not self.no_embedding:
            src = self.src_word_emb(src)
        x = self.pe(src)
        Attn = []
        for i in range(self.N):
            x, attn = self.layers[i](x, mask)
            Attn.append(attn)
        return self.norm(x), Attn



class SimpleEncoderTrue(nn.Module):
    def __init__(self, vocab_size, embedding_dim, N, heads, max_seq_len=200, pe_type=None):
        super().__init__()
        self.N = N
        self.pe = PositionalEncoder(embedding_dim, max_seq_len=max_seq_len, type=pe_type)
        self.layers = get_clones(SimpleEncoderLayerTrue(embedding_dim, heads), N)
        self.norm = Norm(embedding_dim)
        if vocab_size == embedding_dim:
            self.no_embedding = True
        else:
            self.no_embedding = False
            self.src_word_emb = nn.Linear(vocab_size, embedding_dim)

    def forward(self, src, mask=None):
        if not self.no_embedding:
            src = self.src_word_emb(src)
        x = self.pe(src)
        Attn = []
        for i in range(self.N):
            x, attn = self.layers[i](x, mask)
            Attn.append(attn)
        return self.norm(x), Attn


class MixEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, N, heads):
        super().__init__()
        self.N = N
        self.pe = PositionalEncoder(embedding_dim)
        self.layers = get_clones(MixEncoderLayer(embedding_dim, heads), N)
        self.norm = Norm(embedding_dim)
        if vocab_size == embedding_dim:
            self.no_embedding = True
        else:
            self.no_embedding = False
            self.src_word_emb = nn.Linear(vocab_size, embedding_dim)

    def forward(self, src, mask=None):
        if not self.no_embedding:
            src = self.src_word_emb(src)
        x = self.pe(src)
        Attn = []
        for i in range(self.N):
            x, attn = self.layers[i](x, mask)
            Attn.append(attn)
        return self.norm(x), Attn


class MixEncoderNoPE(nn.Module):
    def __init__(self, vocab_size, embedding_dim, N, heads):
        super().__init__()
        self.N = N
        self.layers = get_clones(MixEncoderLayer(embedding_dim, heads), N)
        self.norm = Norm(embedding_dim)

    def forward(self, src, mask=None):
        x = src
        Attn = []
        for i in range(self.N):
            x, attn = self.layers[i](x, mask)
            Attn.append(attn)
        return self.norm(x), Attn


class PEwoPEMixEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, N, heads):
        super().__init__()
        self.N = N
        self.pe = PositionalEncoder(embedding_dim)
        self.layers1 = get_clones(MixEncoderLayer(embedding_dim, heads), N)
        self.layers2 = get_clones(MixEncoderLayer(embedding_dim, heads), N)
        self.norm = Norm(embedding_dim)
        if vocab_size == embedding_dim:
            self.no_embedding = True
        else:
            self.no_embedding = False
            self.src_word_emb = nn.Linear(vocab_size, embedding_dim)

    def forward(self, src, mask=None):
        if not self.no_embedding:
            src = self.src_word_emb(src)
        x = self.pe(src)
        Attn = []
        for i in range(self.N):
            x, attn = self.layers1[i](x, mask)
            Attn.append(attn)
        Attn = []
        y = src
        for i in range(self.N):
            y, attn = self.layers2[i](y, mask)
            Attn.append(attn)
        x = x + y
        return self.norm(x), Attn


class PEwoPEEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, N, heads):
        super().__init__()
        self.N = N
        self.pe = PositionalEncoder(embedding_dim)
        self.layers1 = get_clones(EncoderLayer(embedding_dim, heads), N)
        self.layers2 = get_clones(EncoderLayer(embedding_dim, heads), N)
        self.norm = Norm(embedding_dim)
        if vocab_size == embedding_dim:
            self.no_embedding = True
        else:
            self.no_embedding = False
            self.src_word_emb = nn.Linear(vocab_size, embedding_dim)

    def forward(self, src, mask=None):
        if not self.no_embedding:
            src = self.src_word_emb(src)
        x = self.pe(src)
        Attn = []
        for i in range(self.N):
            x, attn = self.layers1[i](x, mask)
            Attn.append(attn)
        Attn = []
        y = src
        for i in range(self.N):
            y, attn = self.layers2[i](y, mask)
            Attn.append(attn)
        x = x + y
        return self.norm(x), Attn


class PEwoPESimpleEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, N, heads):
        super().__init__()
        self.N = N
        self.pe = PositionalEncoder(embedding_dim)
        self.layers1 = get_clones(SimpleEncoderLayer(embedding_dim, heads), N)
        self.layers2 = get_clones(SimpleEncoderLayer(embedding_dim, heads), N)
        self.norm = Norm(embedding_dim)
        if vocab_size == embedding_dim:
            self.no_embedding = True
        else:
            self.no_embedding = False
            self.src_word_emb = nn.Linear(vocab_size, embedding_dim)

    def forward(self, src, mask=None):
        if not self.no_embedding:
            src = self.src_word_emb(src)
        x = self.pe(src)
        Attn = []
        for i in range(self.N):
            x, attn = self.layers1[i](x, mask)
            Attn.append(attn)
        Attn = []
        y = src
        for i in range(self.N):
            y, attn = self.layers2[i](y, mask)
            Attn.append(attn)
        x = x + y
        return self.norm(x), Attn


class SimpleEncoderNoPE(nn.Module):
    def __init__(self, vocab_size, embedding_dim, N, heads):
        super().__init__()
        self.N = N
        self.layers = get_clones(SimpleEncoderLayer(embedding_dim, heads), N)
        self.norm = Norm(embedding_dim)
        if vocab_size == embedding_dim:
            self.no_embedding = True
        else:
            self.no_embedding = False
            self.src_word_emb = nn.Linear(vocab_size, embedding_dim)

    def forward(self, src, mask=None):
        if not self.no_embedding:
            src = self.src_word_emb(src)
        Attn = []
        x = src
        for i in range(self.N):
            x, attn = self.layers[i](x, mask)
            Attn.append(attn)
        return self.norm(x), Attn


class SimpleEncoderNoPETrue(nn.Module):
    def __init__(self, vocab_size, embedding_dim, N, heads, importance_threshold=0):
        super().__init__()
        self.N = N
        self.layers = get_clones(SimpleEncoderLayerTrue(embedding_dim, heads, importance_threshold), N)
        self.norm = Norm(embedding_dim)
        # if vocab_size == embedding_dim:
        #     self.no_embedding = True
        # else:
        #     self.no_embedding = False
        #     self.src_word_emb = nn.Linear(vocab_size, embedding_dim)

    def forward(self, src, mask=None):
        # if not self.no_embedding:
        #     src = self.src_word_emb(src)
        Attn = []
        x = src
        for i in range(self.N):
            x, attn = self.layers[i](x, mask)
            Attn.append(attn)
        return self.norm(x), Attn


class MostSimpleEncoderNoPE(nn.Module):
    def __init__(self, vocab_size, embedding_dim, N, heads, num_frame=None):
        super().__init__()
        self.N = N
        self.layers = get_clones(MostSimpleEncoderLayer(embedding_dim, heads), N)
        # if num_frame is not None:
        #     self.norm = FrameNorm(vocab_size)
        # else:
        #     self.norm = Norm(embedding_dim)

    def forward(self, src, mask=None):
        Attn = []
        x = src
        for i in range(self.N):
            x, attn = self.layers[i](x, mask)
            Attn.append(attn)
        return x, Attn


class SimpleEncoder2(nn.Module):
    """除了forward函数式的mask的默认值，其他的和SimpleEncoderNoPE完全一样"""

    def __init__(self, vocab_size, embedding_dim, N, heads):
        super().__init__()
        self.N = N
        self.layers = get_clones(SimpleEncoderLayer(embedding_dim, heads), N)
        self.norm = Norm(embedding_dim)
        if vocab_size == embedding_dim:
            self.no_embedding = True
        else:
            self.no_embedding = False
            self.src_word_emb = nn.Linear(vocab_size, embedding_dim)

    def forward(self, src, mask):
        if not self.no_embedding:
            src = self.src_word_emb(src)
        Attn = []
        x = src
        for i in range(self.N):
            x, attn = self.layers[i](x, mask)
            Attn.append(attn)
        return self.norm(x), Attn

class OptimizedSimpleEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, N, heads):
        super().__init__()
        self.N = N
        self.pes = get_clones(PositionalEncoder(embedding_dim))
        self.layers = get_clones(SimpleEncoderLayer(embedding_dim, heads), N)
        self.norm = Norm(embedding_dim)
        if vocab_size == embedding_dim:
            self.no_embedding = True
        else:
            self.no_embedding = False
            self.src_word_emb = nn.Linear(vocab_size, embedding_dim)

    def forward(self, src, mask):
        if not self.no_embedding:
            src = self.src_word_emb(src)
        x = self.pe(src)
        Attn = []
        for i in range(self.N):
            x, attn = self.layers[i](x, mask)
            Attn.append(attn)
        return self.norm(x), Attn


# class PositionalEncoder(nn.Module):
#     def __init__(self, embedding_dim, max_seq_len=200):
#         super().__init__()
#         self.embedding_dim = embedding_dim
#         pe = torch.zeros(max_seq_len, embedding_dim)
#         for pos in range(max_seq_len):
#             for i in range(0, embedding_dim, 2):
#                 pe[pos, i] = \
#                     math.sin(pos / (10000 ** ((2 * i) / embedding_dim)))
#                 pe[pos, i + 1] = \
#                     math.cos(pos / (10000 ** ((2 * (i + 1)) / embedding_dim)))
#         pe = pe.unsqueeze(0)
#         self.register_buffer('pe', pe)
#
#     def forward(self, x):
#         x = x * math.sqrt(self.embedding_dim)
#         seq_len = x.size(1)
#         pe = Variable(self.pe[:, :seq_len], requires_grad=False)
#         if x.is_cuda:
#             pe.cuda()
#         x = x + pe
#         return x

class PositionalEncoder(nn.Module):
    def __init__(self, embedding_dim, max_seq_len=200, type=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        pe = torch.zeros(max_seq_len, embedding_dim)

        # if type is None:
        for pos in range(max_seq_len):
            for i in range(0, embedding_dim, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / embedding_dim)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / embedding_dim)))
        # elif type == "pre":
        #     for pos in range(max_seq_len):
        #         for i in range(0, embedding_dim, 2):
        #             pe[pos, i] = \
        #                 math.sin(pos / (10000 ** ((2 * i) / embedding_dim))) + pos / 1000
        #             pe[pos, i + 1] = \
        #                 math.cos(pos / (10000 ** ((2 * (i + 1)) / embedding_dim))) + pos / 1000
        # elif type == "post":
        #     for pos in range(max_seq_len):
        #         for i in range(0, embedding_dim, 2):
        #             pe[pos, i] = \
        #                 math.sin(pos / (10000 ** ((2 * i) / embedding_dim))) + ((max_seq_len - 1 - pos) / 1000)
        #             pe[pos, i + 1] = \
        #                 math.cos(pos / (10000 ** ((2 * (i + 1)) / embedding_dim))) + ((max_seq_len - 1 - pos) / 1000)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * math.sqrt(self.embedding_dim)
        seq_len = x.size(1)
        pe = Variable(self.pe[:, :seq_len], requires_grad=False)
        if x.is_cuda:
            pe.cuda()
        x = x * pe
        return x


class NewPositionalEncoder(nn.Module):
    def __init__(self, embedding_dim, max_seq_len=200):
        super().__init__()
        self.embedding_dim = embedding_dim
        pe = torch.zeros(max_seq_len, embedding_dim)
        for pos in range(max_seq_len):
            pe[pos, :] = math.sin(((pos / (max_seq_len - 1) / 2) + 0.25) * math.pi)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * math.sqrt(self.embedding_dim)
        seq_len = x.size(1)
        pe = Variable(self.pe[:, :seq_len], requires_grad=True)
        if x.is_cuda:
            pe.cuda()
        x = x + pe
        return x


class Transformer0522(nn.Module):
    """best model on 05/01"""

    def __init__(self, vocab_size, embedding_dim, N, heads, num_class, dropout_rate=0.1):
        super().__init__()
        # self.feature_extractor = nn.Linear(2048, 512)
        self.encoder = Encoder(vocab_size, embedding_dim, N, heads)
        self.sigmoid = nn.Sigmoid()
        self.out = nn.Linear(embedding_dim, num_class + 1)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, src, src_mask):
        # src = self.feature_extractor(src)
        # shape of e_outputs : batchsize, chunksize, size of feature
        output, enc_attn = self.encoder(src, src_mask)
        output = output.mean(dim=1)  # mean on frames in one chunk
        output = self.sigmoid(output)
        output = self.dropout(output)
        output = self.out(output)
        # batch, 4
        return output
        # return output, None, tuple(enc_attn)


class Transformer0522_4_spotting(nn.Module):
    """best model on 05/01"""

    def __init__(self, vocab_size, embedding_dim, N, heads, num_class, dropout_rate=0.1):
        super().__init__()
        # self.feature_extractor = nn.Linear(2048, 512)
        self.encoder = Encoder(vocab_size, embedding_dim, N, heads)
        self.sigmoid = nn.Sigmoid()
        self.out = nn.Linear(embedding_dim, num_class + 1)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, src, src_mask):
        # src = self.feature_extractor(src)
        # shape of e_outputs : batchsize, chunksize, size of feature
        output, enc_attn = self.encoder(src, src_mask)
        output = self.sigmoid(output)
        output = self.dropout(output)
        output = self.out(output)
        output = output.mean(dim=1)  # mean on frames in one chunk
        # batch, 4
        return output
        # return output, None, tuple(enc_attn)


class AllSimpleTransformer(nn.Module):
    """best model on 05/01"""

    def __init__(self, vocab_size, embedding_dim, N, heads, num_class, dropout_rate=0.1):
        super().__init__()
        # self.feature_extractor = nn.Linear(2048, 512)
        self.encoder = SimpleEncoder(vocab_size, embedding_dim, N, heads)
        self.sigmoid = nn.Sigmoid()
        self.out = nn.Linear(embedding_dim, num_class + 1)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, src, src_mask):
        # src = self.feature_extractor(src)
        # shape of e_outputs : batchsize, chunksize, size of feature
        output, enc_attn = self.encoder(src, src_mask)
        output = output.mean(dim=1)  # mean on frames in one chunk
        output = self.sigmoid(output)
        output = self.dropout(output)
        output = self.out(output)
        # batch, 4
        return output
        # return output, None, tuple(enc_attn)


class SimpleTransformer(nn.Module):

    def __init__(self, vocab_size, embedding_dim, N, heads, num_class, dropout_rate=0.1):
        super().__init__()
        self.similar_filter = SimpleEncoder(vocab_size, embedding_dim, 1, heads)
        self.encoder = SimpleEncoder(vocab_size, embedding_dim, N, heads)
        self.sigmoid = nn.Sigmoid()
        self.out = nn.Linear(embedding_dim, num_class + 1)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, src, src_mask):
        output, _ = self.similar_filter(src, None)
        output, enc_attn = self.encoder(output, src_mask)
        output = output.mean(dim=1)  # mean on frames in one chunk
        output = self.sigmoid(output)
        output = self.dropout(output)
        output = self.out(output)
        # batch, 4
        return output


class SimpleTransformer2(nn.Module):

    def __init__(self, vocab_size, embedding_dim, N, heads, num_class, dropout_rate=0.1):
        super().__init__()
        self.similar_filter = SimpleEncoder2(vocab_size, embedding_dim, 1, heads)
        self.encoder = SimpleEncoder(vocab_size, embedding_dim, N, heads)
        self.sigmoid = nn.Sigmoid()
        self.out = nn.Linear(embedding_dim, num_class + 1)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, src, src_mask):
        output, _ = self.similar_filter(src, None)
        output, enc_attn = self.encoder(output, src_mask)
        output = output.mean(dim=1)  # mean on frames in one chunk
        output = self.sigmoid(output)
        output = self.dropout(output)
        output = self.out(output)
        # batch, 4
        return output


class SimpleTransformer2EveryFrame(nn.Module):

    def __init__(self, vocab_size, embedding_dim, N, heads, num_class, dropout_rate=0.1):
        super().__init__()
        self.similar_filter = SimpleEncoder2(vocab_size, embedding_dim, 1, heads)
        self.encoder = SimpleEncoder(vocab_size, embedding_dim, N, heads)
        self.sigmoid = nn.Sigmoid()
        self.out = nn.Linear(embedding_dim, num_class + 1)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, src, src_mask):
        output, _ = self.similar_filter(src, None)
        output, enc_attn = self.encoder(output, src_mask)
        # output = output.mean(dim=1)  # mean on frames in one chunk
        output = self.sigmoid(output)
        output = self.dropout(output)
        output = self.out(output)
        # batch, 4
        return output

class SimpleTransformer3(nn.Module):

    def __init__(self, vocab_size, embedding_dim, N, heads, num_class, dropout_rate=0.1):
        super().__init__()
        self.similar_filter = SimpleEncoder2(vocab_size, embedding_dim, 1, 1)
        self.encoder = SimpleEncoder(vocab_size, embedding_dim, N, heads)
        self.sigmoid = nn.Sigmoid()
        self.out = nn.Linear(embedding_dim, num_class + 1)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, src, src_mask):
        output, _ = self.similar_filter(src, None)
        output, enc_attn = self.encoder(output, src_mask)
        output = output.mean(dim=1)  # mean on frames in one chunk
        output = self.sigmoid(output)
        output = self.dropout(output)
        output = self.out(output)
        # batch, 4
        return output


class PrePostTransformer(nn.Module):

    def __init__(self, vocab_size, embedding_dim, N, heads, num_class, dropout_rate=0.1):
        super().__init__()
        # self.similar_filter = SimpleEncoder2(vocab_size, embedding_dim, 1, 1)
        self.pre_encoder = SimpleEncoder(vocab_size, embedding_dim, N, heads)
        self.post_encoder = SimpleEncoder(vocab_size, embedding_dim, N, heads)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.out = nn.Linear(embedding_dim, num_class + 1)

    def forward(self, src, src_mask):
        # output, _ = self.similar_filter(src, None)
        pre_output, enc_attn = self.pre_encoder(src, src_mask)
        pre_output = pre_output.mean(dim=1)  # mean on frames in one chunk
        pre_output = self.sigmoid(pre_output)
        pre_output = self.dropout(pre_output)
        pre_output = self.out(pre_output)

        post_output, enc_attn = self.post_encoder(src, src_mask)
        post_output = post_output.mean(dim=1)  # mean on frames in one chunk
        post_output = self.sigmoid(post_output)
        post_output = self.dropout(post_output)
        post_output = self.out(post_output)
        # batch, 4
        return pre_output, post_output


class E2ETransformer(nn.Module):

    def __init__(self, vocab_size, embedding_dim, N, heads, num_class, dropout_rate=0.1, similar_threshold=0.3):
        super().__init__()
        self.similar_filter = SimpleEncoder(vocab_size, embedding_dim, 1, heads)
        self.encoder = SimpleEncoder(vocab_size, embedding_dim, N, heads)
        self.sigmoid = nn.Sigmoid()
        self.out1 = nn.Linear(embedding_dim, num_class + 1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.out2 = nn.Linear(embedding_dim, num_class + 1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

        self.similar_threshold = similar_threshold

    def forward(self, src, src_mask):
        output_1, _ = self.similar_filter(src, None)

        output, enc_attn = self.encoder(src, src_mask)
        output = output.mean(dim=1)  # mean on frames in one chunk
        output = self.sigmoid(output)
        output = self.dropout1(output)
        output = self.out1(output)

        output_1 = output_1.mean(dim=1)  # mean on frames in one chunk
        output_1 = self.sigmoid(output_1)
        output_1 = self.dropout2(output_1)
        output_1 = self.out2(output_1)

        # batch, 4
        return output_1, output


class MyTransformer(nn.Module):

    def __init__(self, vocab_size, embedding_dim, n_layer, n_head, num_class, args):
        super().__init__()
        # self.transformer = Transformer(vocab_size, embedding_dim, N, heads)
        # encoder_layer = TransformerEncoderLayer(vocab_size, n_head, embedding_dim, 0.1, 'relu')
        # encoder_norm = LayerNorm(vocab_size)
        # self.encoder_4_classify = TransformerEncoder(encoder_layer, n_layer, encoder_norm)
        # self.encoder_4_spot = TransformerEncoder(encoder_layer, n_layer, encoder_norm)
        self.encoder_4_spot = Transformer0522(vocab_size, embedding_dim, n_layer, n_head,
                                              args.chunk_size * args.framerate - 1, dropout_rate=args.dropout_rate)
        self.encoder_4_classify = Transformer0522(vocab_size, embedding_dim, n_layer, n_head, num_class,
                                                  dropout_rate=args.dropout_rate)
        self.filter = BaseFilter(args.similar_threshold, n_important_frames=args.n_important_frames,
                                 pooling_module=args.pooling_module)
        # self.classify_linear_0 = nn.Linear(embedding_dim, num_class + 1)
        # self.time_spotting_linear_0 = nn.Linear(embedding_dim, 64)
        # self.time_spotting_linear_1 = nn.Linear(64, 1)
        self.similar_threshold = args.similar_threshold
        self.use_spot_loss = args.use_spot_loss
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # self.sigmoid = nn.Sigmoid()
        # self.dropout = nn.Dropout(p=0.1)

    def forward(self, src, similarity, src_mask=None):
        if self.similar_threshold == 0:
            filted_output = self.encoder_4_classify(src, src_mask)
        else:
            filted_src = self.filter(src, similarity)
            if len(filted_src) == 0:
                filted_output = self.encoder_4_classify(src, src_mask)
            else:
                filted_output = []
                for item in filted_src:
                    temp_output = self.encoder_4_classify(item.unsqueeze(0), src_mask)
                    # temp_output = torch.mean(temp_output, dim=1)
                    filted_output.append(temp_output)
                filted_output = torch.stack(filted_output)
        # output_0 = self.classify_linear_0(filted_output)
        if self.use_spot_loss:
            output = self.encoder_4_spot(src, src_mask)
            return filted_output.squeeze(dim=1), output
        else:
            return filted_output.squeeze(dim=1), None


class AllSimpleMyTransformer(nn.Module):

    def __init__(self, vocab_size, embedding_dim, n_layer, n_head, num_class, args):
        super().__init__()
        self.encoder_4_spot = AllSimpleTransformer(vocab_size, embedding_dim, n_layer, n_head,
                                                   args.chunk_size * args.framerate - 1,
                                                   dropout_rate=args.dropout_rate)
        self.encoder_4_classify = AllSimpleTransformer(vocab_size, embedding_dim, n_layer, n_head, num_class,
                                                       dropout_rate=args.dropout_rate)
        self.filter = BaseFilter(args.similar_threshold, n_important_frames=args.n_important_frames,
                                 pooling_module=args.pooling_module)
        self.similar_threshold = args.similar_threshold
        self.use_spot_loss = args.use_spot_loss
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, similarity, src_mask=None):
        if self.similar_threshold == 0:
            filted_output = self.encoder_4_classify(src, src_mask)
        else:
            filted_src = self.filter(src, similarity)
            if len(filted_src) == 0:
                filted_output = self.encoder_4_classify(src, src_mask)
            else:
                filted_output = []
                for item in filted_src:
                    temp_output = self.encoder_4_classify(item.unsqueeze(0), src_mask)
                    filted_output.append(temp_output)
                filted_output = torch.stack(filted_output)
        if self.use_spot_loss:
            output = self.encoder_4_spot(src, src_mask)
            return filted_output.squeeze(dim=1), output
        else:
            return filted_output.squeeze(dim=1), None


class SimpleMyTransformer(nn.Module):

    def __init__(self, vocab_size, embedding_dim, n_layer, n_head, num_class, args):
        super().__init__()
        self.encoder_4_spot = AllSimpleTransformer(vocab_size, embedding_dim, n_layer, n_head,
                                                   args.chunk_size * args.framerate - 1,
                                                   dropout_rate=args.dropout_rate)
        self.encoder_4_classify = AllSimpleTransformer(vocab_size, embedding_dim, n_layer, n_head, num_class,
                                                       dropout_rate=args.dropout_rate)
        self.filter = BaseFilter(args.similar_threshold, n_important_frames=args.n_important_frames,
                                 pooling_module=args.pooling_module)
        self.similar_threshold = args.similar_threshold
        self.use_spot_loss = args.use_spot_loss
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, similarity, src_mask=None):
        if self.similar_threshold == 0:
            filted_output = self.encoder_4_classify(src, src_mask)
        else:
            filted_src = self.filter(src, similarity)
            if len(filted_src) == 0:
                filted_output = self.encoder_4_classify(src, src_mask)
            else:
                filted_output = []
                for item in filted_src:
                    temp_output = self.encoder_4_classify(item.unsqueeze(0), src_mask)
                    filted_output.append(temp_output)
                filted_output = torch.stack(filted_output)
        if self.use_spot_loss:
            output = self.encoder_4_spot(src, src_mask)
            return filted_output.squeeze(dim=1), output
        else:
            return filted_output.squeeze(dim=1), None


class MyOptimizeTransformer(nn.Module):
    """第一个encoder层没有全连接"""

    def __init__(self, vocab_size, embedding_dim, n_layer, n_head, num_class, args):
        super().__init__()
        self.encoder_4_spot = SimpleTransformer(vocab_size, embedding_dim, n_layer, n_head,
                                                args.chunk_size * args.framerate - 1,
                                                dropout_rate=args.dropout_rate)
        self.encoder_4_classify = SimpleTransformer(vocab_size, embedding_dim, n_layer, n_head, num_class,
                                                    dropout_rate=args.dropout_rate)
        self.filter = BaseFilter(args.similar_threshold, n_important_frames=args.n_important_frames,
                                 pooling_module=args.pooling_module)
        self.similar_threshold = args.similar_threshold
        self.use_spot_loss = args.use_spot_loss
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, similarity, src_mask=None):
        if self.similar_threshold == 0:
            filted_output = self.encoder_4_classify(src, src_mask)
        else:
            filted_src = self.filter(src, similarity)
            if len(filted_src) == 0:
                filted_output = self.encoder_4_classify(src, src_mask)
            else:
                filted_output = []
                for item in filted_src:
                    temp_output = self.encoder_4_classify(item.unsqueeze(0), src_mask)
                    filted_output.append(temp_output)
                filted_output = torch.stack(filted_output)
        if self.use_spot_loss:
            output = self.encoder_4_spot(src, src_mask)
            return filted_output.squeeze(dim=1), output
        else:
            return filted_output.squeeze(dim=1), None


class MyOptimizeTransformer2(nn.Module):
    """第一个encoder层没有全连接"""

    def __init__(self, vocab_size, embedding_dim, n_layer, n_head, num_class, args):
        super().__init__()
        self.encoder_4_spot = SimpleTransformer(vocab_size, embedding_dim, n_layer, n_head,
                                                args.chunk_size * args.framerate - 1,
                                                dropout_rate=args.dropout_rate)
        self.encoder_4_classify = Transformer0522(vocab_size, embedding_dim, n_layer, n_head, num_class,
                                                    dropout_rate=args.dropout_rate)
        self.filter = BaseFilter(args.similar_threshold, n_important_frames=args.n_important_frames,
                                 pooling_module=args.pooling_module)
        self.similar_threshold = args.similar_threshold
        self.use_spot_loss = args.use_spot_loss
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, similarity, src_mask=None):
        if self.similar_threshold == 0:
            filted_output = self.encoder_4_classify(src, src_mask)
        else:
            filted_src = self.filter(src, similarity)
            if len(filted_src) == 0:
                filted_output = self.encoder_4_classify(src, src_mask)
            else:
                filted_output = []
                for item in filted_src:
                    temp_output = self.encoder_4_classify(item.unsqueeze(0), src_mask)
                    filted_output.append(temp_output)
                filted_output = torch.stack(filted_output)
        if self.use_spot_loss:
            output = self.encoder_4_spot(src, src_mask)
            return filted_output.squeeze(dim=1), output
        else:
            return filted_output.squeeze(dim=1), None


class MyOptimizeTransformer3(nn.Module):
    """第一个encoder层没有全连接"""

    def __init__(self, vocab_size, embedding_dim, n_layer, n_head, num_class, args):
        super().__init__()
        self.encoder_4_spot = SimpleTransformer2(vocab_size, embedding_dim, n_layer, n_head,
                                                args.chunk_size * args.framerate - 1,
                                                dropout_rate=args.dropout_rate)
        self.encoder_4_classify = SimpleTransformer2(vocab_size, embedding_dim, n_layer, n_head, num_class,
                                                    dropout_rate=args.dropout_rate)
        self.filter = BaseFilter(args.similar_threshold, n_important_frames=args.n_important_frames,
                                 pooling_module=args.pooling_module)
        self.similar_threshold = args.similar_threshold
        self.use_spot_loss = args.use_spot_loss
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, similarity, src_mask=None):
        if self.similar_threshold == 0:
            filted_output = self.encoder_4_classify(src, src_mask)
        else:
            filted_src = self.filter(src, similarity)
            if len(filted_src) == 0:
                filted_output = self.encoder_4_classify(src, src_mask)
            else:
                filted_output = []
                for item in filted_src:
                    temp_output = self.encoder_4_classify(item.unsqueeze(0), src_mask)
                    filted_output.append(temp_output)
                filted_output = torch.stack(filted_output)
        if self.use_spot_loss:
            output = self.encoder_4_spot(src, src_mask)
            return filted_output.squeeze(dim=1), output
        else:
            return filted_output.squeeze(dim=1), None


class MyOptimizeTransformer4(nn.Module):
    """第一个encoder层没有全连接"""

    def __init__(self, vocab_size, embedding_dim, n_layer, n_head, num_class, args):
        super().__init__()
        self.encoder_4_spot = SimpleTransformer2(vocab_size, embedding_dim, n_layer, n_head,
                                                args.chunk_size * args.framerate - 1,
                                                dropout_rate=args.dropout_rate)
        self.encoder_4_classify = SimpleTransformer2EveryFrame(vocab_size, embedding_dim, n_layer, n_head, num_class,
                                                    dropout_rate=args.dropout_rate)
        self.filter = BaseFilter(args.similar_threshold, n_important_frames=args.n_important_frames,
                                 pooling_module=args.pooling_module)
        self.similar_threshold = args.similar_threshold
        self.use_spot_loss = args.use_spot_loss
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, similarity, src_mask=None):
        if self.similar_threshold == 0:
            filted_output = self.encoder_4_classify(src, src_mask)
        else:
            filted_src = self.filter(src, similarity)
            if len(filted_src) == 0:
                filted_output = self.encoder_4_classify(src, src_mask)
            else:
                filted_output = []
                for item in filted_src:
                    temp_output = self.encoder_4_classify(item.unsqueeze(0), src_mask)
                    filted_output.append(temp_output)
                filted_output = torch.stack(filted_output)
        if self.use_spot_loss:
            output = self.encoder_4_spot(src, src_mask)
            return filted_output.squeeze(dim=1), output
        else:
            return filted_output.squeeze(dim=1), None


class MyOptimizeTransformer5(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_layer, n_head, num_class, args):
        super().__init__()
        self.encoder_4_spot = SimpleTransformer3(vocab_size, embedding_dim, n_layer, n_head,
                                                args.chunk_size * args.framerate - 1,
                                                dropout_rate=args.dropout_rate)
        self.encoder_4_classify = SimpleTransformer3(vocab_size, embedding_dim, n_layer, n_head, num_class,
                                                    dropout_rate=args.dropout_rate)
        self.filter = BaseFilter(args.similar_threshold, n_important_frames=args.n_important_frames,
                                 pooling_module=args.pooling_module)
        self.similar_threshold = args.similar_threshold
        self.use_spot_loss = args.use_spot_loss
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, similarity, src_mask=None):
        if self.similar_threshold == 0:
            filted_output = self.encoder_4_classify(src, src_mask)
        else:
            filted_src = self.filter(src, similarity)
            if len(filted_src) == 0:
                filted_output = self.encoder_4_classify(src, src_mask)
            else:
                filted_output = []
                for item in filted_src:
                    temp_output = self.encoder_4_classify(item.unsqueeze(0), src_mask)
                    filted_output.append(temp_output)
                filted_output = torch.stack(filted_output)
        if self.use_spot_loss:
            output = self.encoder_4_spot(src, src_mask)
            return filted_output.squeeze(dim=1), output
        else:
            return filted_output.squeeze(dim=1), None


class MyOptimizeTransformer6(nn.Module):
    # 分层的transofmer
    def __init__(self, vocab_size, embedding_dim, n_layer, n_head, num_class, args):
        super().__init__()
        self.encoder_4_classify = PrePostTransformer(vocab_size, embedding_dim, n_layer, n_head, num_class,
                                                     dropout_rate=args.dropout_rate)
        self.encoder_4_classify1_1 = PrePostTransformer(vocab_size, embedding_dim, n_layer, n_head, num_class,
                                                        dropout_rate=args.dropout_rate)
        self.encoder_4_classify1_2 = PrePostTransformer(vocab_size, embedding_dim, n_layer, n_head, num_class,
                                                        dropout_rate=args.dropout_rate)

        self.use_spot_loss = args.use_spot_loss
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, similarity=None, src_mask=None):
        b, n_frame, f = src.shape
        pre_output, post_output = self.encoder_4_classify(src, src_mask)
        pre_output1_1, post_output1_1 = self.encoder_4_classify1_1(src[:, :n_frame // 2, :], src_mask)
        pre_output1_2, post_output1_2 = self.encoder_4_classify1_2(src[:, n_frame // 2:, :], src_mask)

        return pre_output, post_output, pre_output1_1, post_output1_1, pre_output1_2, post_output1_2


class MyOptimizeTransformer7(nn.Module):
    # 分层的transofmer
    def __init__(self, vocab_size, embedding_dim, n_layer, n_head, num_class, args):
        super().__init__()
        self.pre_encoder = SimpleEncoder(vocab_size, embedding_dim, n_layer, n_head)
        self.post_encoder = SimpleEncoder(vocab_size, embedding_dim, n_layer, n_head)
        self.pre_encoder11 = SimpleEncoderNoPE(vocab_size, embedding_dim, n_layer, n_head)
        self.post_encoder11 = SimpleEncoderNoPE(vocab_size, embedding_dim, n_layer, n_head)
        self.pre_encoder12 = SimpleEncoderNoPE(vocab_size, embedding_dim, n_layer, n_head)
        self.post_encoder12 = SimpleEncoderNoPE(vocab_size, embedding_dim, n_layer, n_head)
        self.sigmoid1 = nn.Sigmoid()
        self.out1 = nn.Linear(embedding_dim, num_class + 1)
        self.dropout1 = nn.Dropout(p=args.dropout_rate)

        self.sigmoid2 = nn.Sigmoid()
        self.out2 = nn.Linear(embedding_dim, num_class + 1)
        self.dropout2 = nn.Dropout(p=args.dropout_rate)

        self.use_spot_loss = args.use_spot_loss
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, similarity=None, src_mask=None):
        b, n_frame, f = src.shape
        pre_content = self.pre_encoder(src)[0][:, :n_frame // 2]
        post_content = self.post_encoder(src)[0][:, n_frame // 2:]
        pre_output1_1 = self.pre_encoder11(pre_content)[0][:, :n_frame // 4]
        post_output1_1 = self.post_encoder11(pre_content)[0][:, n_frame // 4:]
        pre_output1_2 = self.pre_encoder12(post_content)[0][:, :n_frame // 4]
        post_output1_2 = self.post_encoder12(post_content)[0][:, n_frame // 4:]

        pre_output = pre_content.mean(dim=1)
        pre_output = self.sigmoid1(pre_output)
        pre_output = self.dropout1(pre_output)
        pre_output = self.out1(pre_output)

        post_output = post_content.mean(dim=1)
        post_output = self.sigmoid1(post_output)
        post_output = self.dropout1(post_output)
        post_output = self.out1(post_output)

        pre_output1_1 = pre_output1_1.mean(dim=1)
        pre_output1_1 = self.sigmoid2(pre_output1_1)
        pre_output1_1 = self.dropout2(pre_output1_1)
        pre_output1_1 = self.out2(pre_output1_1)

        post_output1_1 = post_output1_1.mean(dim=1)
        post_output1_1 = self.sigmoid2(post_output1_1)
        post_output1_1 = self.dropout2(post_output1_1)
        post_output1_1 = self.out2(post_output1_1)

        pre_output1_2 = pre_output1_2.mean(dim=1)
        pre_output1_2 = self.sigmoid2(pre_output1_2)
        pre_output1_2 = self.dropout2(pre_output1_2)
        pre_output1_2 = self.out2(pre_output1_2)

        post_output1_2 = post_output1_2.mean(dim=1)
        post_output1_2 = self.sigmoid2(post_output1_2)
        post_output1_2 = self.dropout2(post_output1_2)
        post_output1_2 = self.out2(post_output1_2)

        return pre_output, post_output, pre_output1_1, post_output1_1, pre_output1_2, post_output1_2


class MyOptimizeTransformer8(nn.Module):
    # 分层的transofmer
    def __init__(self, vocab_size, embedding_dim, n_layer, n_head, num_class, args):
        super().__init__()
        self.pre_encoder = SimpleEncoder(vocab_size, embedding_dim, n_layer, n_head)
        self.post_encoder = SimpleEncoder(vocab_size, embedding_dim, n_layer, n_head)
        self.pre_encoder11 = SimpleEncoderNoPE(vocab_size, embedding_dim, n_layer, n_head)
        self.post_encoder11 = SimpleEncoderNoPE(vocab_size, embedding_dim, n_layer, n_head)
        self.pre_encoder12 = SimpleEncoderNoPE(vocab_size, embedding_dim, n_layer, n_head)
        self.post_encoder12 = SimpleEncoderNoPE(vocab_size, embedding_dim, n_layer, n_head)
        self.sigmoid1 = nn.Sigmoid()
        self.out1 = nn.Linear(embedding_dim, num_class + 1)
        self.dropout1 = nn.Dropout(p=args.dropout_rate)

        self.sigmoid2 = nn.Sigmoid()
        self.out2 = nn.Linear(embedding_dim, num_class + 1)
        self.dropout2 = nn.Dropout(p=args.dropout_rate)

        self.use_spot_loss = args.use_spot_loss
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, src, similarity=None, src_mask=None):
        b, n_frame, f = src.shape
        src_1, src_2, src_3, src_4 = src[:, :n_frame // 4], src[:, n_frame // 4:n_frame // 2], src[:,
                                                                                               n_frame // 2:-n_frame // 4], src[
                                                                                                                            :,
                                                                                                                            -n_frame // 4:]
        pre_output1_1 = self.pre_encoder11(src_1)[0]
        post_output1_1 = self.post_encoder11(src_2)[0]
        pre_output1_2 = self.pre_encoder12(src_3)[0]
        post_output1_2 = self.post_encoder12(src_4)[0]
        pre_content = torch.cat((pre_output1_1, post_output1_1),dim=1)
        pre_content = self.pre_encoder(src[:, :n_frame // 2] + pre_content)[0]
        post_content = torch.cat((pre_output1_2, post_output1_2),dim=1)
        post_content = self.post_encoder(src[:, n_frame // 2:] + post_content)[0]

        pre_output = pre_content.mean(dim=1)
        pre_output = self.sigmoid1(pre_output)
        pre_output = self.dropout1(pre_output)
        pre_output = self.out1(pre_output)

        post_output = post_content.mean(dim=1)
        post_output = self.sigmoid1(post_output)
        post_output = self.dropout1(post_output)
        post_output = self.out1(post_output)

        pre_output1_1 = pre_output1_1.mean(dim=1)
        pre_output1_1 = self.sigmoid2(pre_output1_1)
        pre_output1_1 = self.dropout2(pre_output1_1)
        pre_output1_1 = self.out2(pre_output1_1)

        post_output1_1 = post_output1_1.mean(dim=1)
        post_output1_1 = self.sigmoid2(post_output1_1)
        post_output1_1 = self.dropout2(post_output1_1)
        post_output1_1 = self.out2(post_output1_1)

        pre_output1_2 = pre_output1_2.mean(dim=1)
        pre_output1_2 = self.sigmoid2(pre_output1_2)
        pre_output1_2 = self.dropout2(pre_output1_2)
        pre_output1_2 = self.out2(pre_output1_2)

        post_output1_2 = post_output1_2.mean(dim=1)
        post_output1_2 = self.sigmoid2(post_output1_2)
        post_output1_2 = self.dropout2(post_output1_2)
        post_output1_2 = self.out2(post_output1_2)

        return pre_output, post_output, pre_output1_1, post_output1_1, pre_output1_2, post_output1_2


class MyOptimizeTransformer9(nn.Module):
    # 分层的transofmer
    def __init__(self, vocab_size, embedding_dim, n_layer, n_head, num_class, args):
        super().__init__()
        self.pre_encoder = SimpleEncoder(vocab_size, embedding_dim, n_layer, n_head)
        self.post_encoder = SimpleEncoder(vocab_size, embedding_dim, n_layer, n_head)
        self.pre_encoder11 = SimpleEncoderNoPE(vocab_size, embedding_dim, n_layer, n_head)
        self.post_encoder11 = SimpleEncoderNoPE(vocab_size, embedding_dim, n_layer, n_head)
        self.pre_encoder12 = SimpleEncoderNoPE(vocab_size, embedding_dim, n_layer, n_head)
        self.post_encoder12 = SimpleEncoderNoPE(vocab_size, embedding_dim, n_layer, n_head)
        self.sigmoid1 = nn.Sigmoid()
        self.out1 = nn.Linear(embedding_dim, num_class + 1)
        self.dropout1 = nn.Dropout(p=args.dropout_rate)

        self.sigmoid2 = nn.Sigmoid()
        self.out2 = nn.Linear(embedding_dim, num_class + 1)
        self.dropout2 = nn.Dropout(p=args.dropout_rate)

        self.use_spot_loss = args.use_spot_loss
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, similarity=None, src_mask=None):
        b, n_frame, f = src.shape
        src_1, src_2, src_3, src_4 = src[:, :n_frame // 4], src[:, n_frame // 4:n_frame // 2], src[:,
                                                                                               n_frame // 2:-n_frame // 4], src[
                                                                                                                            :,
                                                                                                                            -n_frame // 4:]
        pre_output1_1 = self.pre_encoder11(src_1)[0]
        post_output1_1 = self.post_encoder11(src_2)[0]
        pre_output1_2 = self.pre_encoder12(src_3)[0]
        post_output1_2 = self.post_encoder12(src_4)[0]
        pre_content = torch.cat((pre_output1_1, post_output1_1), dim=1)
        pre_output = self.pre_encoder(src[:, :n_frame // 2] + pre_content)[0]
        post_content = torch.cat((pre_output1_2, post_output1_2), dim=1)
        post_output = self.post_encoder(src[:, n_frame // 2:] + post_content)[0]

        pre_output = self.sigmoid1(pre_output)
        pre_output = self.dropout1(pre_output)
        pre_output = self.out1(pre_output)

        post_output = self.sigmoid1(post_output)
        post_output = self.dropout1(post_output)
        post_output = self.out1(post_output)

        pre_output1_1 = self.sigmoid2(pre_output1_1)
        pre_output1_1 = self.dropout2(pre_output1_1)
        pre_output1_1 = self.out2(pre_output1_1)

        post_output1_1 = self.sigmoid2(post_output1_1)
        post_output1_1 = self.dropout2(post_output1_1)
        post_output1_1 = self.out2(post_output1_1)

        pre_output1_2 = self.sigmoid2(pre_output1_2)
        pre_output1_2 = self.dropout2(pre_output1_2)
        pre_output1_2 = self.out2(pre_output1_2)

        post_output1_2 = self.sigmoid2(post_output1_2)
        post_output1_2 = self.dropout2(post_output1_2)
        post_output1_2 = self.out2(post_output1_2)
        pre_output[:, :n_frame // 4] += pre_output1_1
        pre_output[:, n_frame // 4:] += post_output1_1
        post_output[:, :n_frame // 4] += pre_output1_2
        post_output[:, n_frame // 4:] += post_output1_2
        output = torch.cat((pre_output, post_output), dim=1)
        # output = 2 * pre_output + 2 * post_output + pre_output1_1 + post_output1_1 + pre_output1_2 + post_output1_2
        return output, None
        # return pre_output, post_output, pre_output1_1, post_output1_1, pre_output1_2, post_output1_2


#
# class MyOptimizeTransformer61(nn.Module):
#     # 分层的transofmer
#     def __init__(self, vocab_size, embedding_dim, n_layer, n_head, num_class, args):
#         super().__init__()
#         self.encoder_4_classify = PrePostTransformer(vocab_size, embedding_dim, n_layer, n_head, num_class,
#                                                      dropout_rate=args.dropout_rate)
#         self.encoder_4_classify1_1 = PrePostTransformer(vocab_size, embedding_dim, n_layer, n_head, num_class,
#                                                         dropout_rate=args.dropout_rate)
#         self.encoder_4_classify1_2 = PrePostTransformer(vocab_size, embedding_dim, n_layer, n_head, num_class,
#                                                         dropout_rate=args.dropout_rate)
#
#         self.use_spot_loss = args.use_spot_loss
#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)
#
#     def forward(self, src, similarity=None, src_mask=None):
#         b, n_frame, f = src.shape
#         pre_output, post_output = self.encoder_4_classify(src, src_mask)
#         pre_output1_1, post_output1_1 = self.encoder_4_classify1_1(src, src_mask)
#         pre_output1_2, post_output1_2 = self.encoder_4_classify1_2(src, src_mask)
#
#         return pre_output, post_output, pre_output1_1, post_output1_1, pre_output1_2, post_output1_2


class MyOptimizeTransformer61(nn.Module):
    # 分层的transofmer
    def __init__(self, vocab_size, embedding_dim, n_layer, n_head, num_class, args):
        super().__init__()
        self.pre_encoder = SimpleEncoder(vocab_size, embedding_dim, n_layer, n_head)
        self.post_encoder = SimpleEncoder(vocab_size, embedding_dim, n_layer, n_head)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=args.dropout_rate)
        self.out = nn.Linear(embedding_dim, num_class + 1)
        self.encoder_4_classify = PrePostTransformer(vocab_size, embedding_dim, n_layer, n_head, num_class,
                                                     dropout_rate=args.dropout_rate)
        self.encoder_4_classify1_1 = PrePostTransformer(vocab_size, embedding_dim, n_layer, n_head, num_class,
                                                        dropout_rate=args.dropout_rate)
        self.encoder_4_classify1_2 = PrePostTransformer(vocab_size, embedding_dim, n_layer, n_head, num_class,
                                                        dropout_rate=args.dropout_rate)

        self.use_spot_loss = args.use_spot_loss
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, similarity=None, src_mask=None):
        b, n_frame, f = src.shape
        pre_output, post_output = self.encoder_4_classify(src, src_mask)
        pre_output1_1, post_output1_1 = self.encoder_4_classify1_1(src, src_mask)
        pre_output1_2, post_output1_2 = self.encoder_4_classify1_2(src, src_mask)

        return pre_output, post_output, pre_output1_1, post_output1_1, pre_output1_2, post_output1_2


class MyOptimizeTransformer51(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_layer, n_head, num_class, args):
        super().__init__()
        # self.encoder_4_spot = SimpleTransformer3(vocab_size, embedding_dim, n_layer, n_head,
        #                                         args.chunk_size * args.framerate - 1,
        #                                         dropout_rate=args.dropout_rate)
        self.encoder_4_classify = SimpleTransformer3(vocab_size, embedding_dim, n_layer, n_head, num_class,
                                                    dropout_rate=args.dropout_rate)
        self.filter = BaseFilter(args.similar_threshold, n_important_frames=args.n_important_frames,
                                 pooling_module=args.pooling_module)
        self.similar_threshold = args.similar_threshold
        self.use_spot_loss = args.use_spot_loss
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, similarity, src_mask=None):
        if self.similar_threshold == 0:
            filted_output = self.encoder_4_classify(src, src_mask)
        else:
            filted_src = self.filter(src, similarity)
            if len(filted_src) == 0:
                filted_output = self.encoder_4_classify(src, src_mask)
            else:
                filted_output = []
                for item in filted_src:
                    temp_output = self.encoder_4_classify(item.unsqueeze(0), src_mask)
                    filted_output.append(temp_output)
                filted_output = torch.stack(filted_output)
        # if self.use_spot_loss:
        #     output = self.encoder_4_spot(src, src_mask)
        #     return filted_output.squeeze(dim=1), output
        # else:
        return filted_output.squeeze(dim=1), None

class MyE2ETransformer(nn.Module):

    def __init__(self, vocab_size, embedding_dim, n_layer, n_head, num_class, args):
        super().__init__()
        if args.use_spot_loss:
            self.encoder_4_spot = SimpleTransformer(vocab_size, embedding_dim, n_layer, n_head,
                                                    args.chunk_size * args.framerate - 1,
                                                    dropout_rate=args.dropout_rate)
        self.encoder_4_classify = E2ETransformer(vocab_size, embedding_dim, n_layer, n_head, num_class,
                                                 dropout_rate=args.dropout_rate)
        if args.similar_threshold > 0:
            self.filter = BaseFilter(args.similar_threshold, n_important_frames=args.n_important_frames,
                                     pooling_module=args.pooling_module)

        self.similar_threshold = args.similar_threshold
        self.use_spot_loss = args.use_spot_loss
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, similarity, src_mask=None):
        if self.similar_threshold == 0:
            filted_output1, filted_output2 = self.encoder_4_classify(src, src_mask)
        else:
            filted_src = self.filter(src, similarity, similar_threshold=self.similar_threshold)
            if len(filted_src) == 0:
                filted_output1, filted_output2 = self.encoder_4_classify(src, src_mask)
            else:
                filted_output1 = []
                filted_output2 = []
                for item in filted_src:
                    temp_output1, temp_output2 = self.encoder_4_classify(item.unsqueeze(0), src_mask)
                    filted_output1.append(temp_output1)
                    filted_output2.append(temp_output2)
                filted_output1 = torch.stack(filted_output1)
                filted_output2 = torch.stack(filted_output2)
        if self.use_spot_loss:
            output = self.encoder_4_spot(src, src_mask)
            return filted_output1.squeeze(dim=1), output, filted_output2.squeeze(dim=1),
        else:
            return filted_output1.squeeze(dim=1), None, filted_output2.squeeze(dim=1),


class MyBaseTransformer(nn.Module):

    def __init__(self, vocab_size, embedding_dim, n_layer, n_head, num_class, args):
        super().__init__()
        self.encoder_4_classify = Transformer0522(vocab_size, embedding_dim, n_layer, n_head, num_class)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_mask=None):
        return self.encoder_4_classify(src, src_mask), None


class NetVLAD2(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=64, dim=128,
                 normalize_input=True, vladv2=False):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
            vladv2 : bool
                If true, use vladv2 otherwise use vladv1
        """
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = 0
        self.vladv2 = vladv2
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=vladv2)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))

    def forward(self, x):
        N, C = x.shape[:2]

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        # soft-assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        x_flatten = x.view(N, C, -1)

        # calculate residuals to each clusters
        vlad = torch.zeros([N, self.num_clusters, C], dtype=x.dtype, layout=x.layout, device=x.device)
        for C in range(self.num_clusters):  # slower than non-looped, but lower memory usage
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - \
                       self.centroids[C:C + 1, :].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
            residual *= soft_assign[:, C:C + 1, :].unsqueeze(2)
            vlad[:, C:C + 1, :] = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        return vlad


class NetVLAD(nn.Module):
    def __init__(self, cluster_size, feature_size, add_batch_norm=True):
        super(NetVLAD, self).__init__()
        self.feature_size = feature_size
        self.cluster_size = cluster_size
        self.clusters = nn.Parameter((1 / math.sqrt(feature_size))
                                     * torch.randn(feature_size, cluster_size))
        self.clusters2 = nn.Parameter((1 / math.sqrt(feature_size))
                                      * torch.randn(1, feature_size, cluster_size))

        self.add_batch_norm = add_batch_norm
        self.out_dim = cluster_size * feature_size

    def forward(self, x):
        # x [BS, T, D]
        max_sample = x.size()[1]

        # LOUPE
        if self.add_batch_norm:  # normalization along feature dimension
            x = F.normalize(x, p=2, dim=2)

        x = x.reshape(-1, self.feature_size)
        assignment = torch.matmul(x, self.clusters)

        assignment = F.softmax(assignment, dim=1)
        assignment = assignment.view(-1, max_sample, self.cluster_size)

        a_sum = torch.sum(assignment, -2, keepdim=True)
        a = a_sum * self.clusters2

        assignment = assignment.transpose(1, 2)

        x = x.view(-1, max_sample, self.feature_size)
        vlad = torch.matmul(assignment, x)
        vlad = vlad.transpose(1, 2)
        vlad = vlad - a

        # L2 intra norm
        vlad = F.normalize(vlad)

        # flattening + L2 norm
        vlad = vlad.reshape(-1, self.cluster_size * self.feature_size)
        vlad = F.normalize(vlad)

        return vlad.squeeze()


class FilterTypeEnum(Enum):
    NetVLAD = "NetVLAD"
    Mean = "Mean"
    Random = "Random"
    Max = "Max"
    Attention = "Attention"
    Transformer = "Transformer"
    First = "First"
    Last = "Last"
    Middle = "Middle"


class BaseFilter(nn.Module):

    def __init__(self, similar_threshold=0.1, n_important_frames=None, pooling_module=FilterTypeEnum.NetVLAD):
        super().__init__()
        self.pooling_module = pooling_module
        self.similar_threshold = similar_threshold
        self.n_important_frames = n_important_frames
        if self.n_important_frames is not None:
            self.main_forward = self.topk_similar
        else:
            self.main_forward = self.threshold_similar

        if self.pooling_module == FilterTypeEnum.NetVLAD.value:
            self.pooling_module = NetVLAD(cluster_size=1, feature_size=512)
            self.pooling = self.netvlad_pooling
        elif self.pooling_module == FilterTypeEnum.Mean.value:
            self.pooling = self.mean_pooling
        elif self.pooling_module == FilterTypeEnum.Max.value:
            self.pooling = self.max_pooling
        elif self.pooling_module == FilterTypeEnum.First.value:
            self.pooling = self.first_pooling
        elif self.pooling_module == FilterTypeEnum.Last.value:
            self.pooling = self.last_pooling
        elif self.pooling_module == FilterTypeEnum.Random.value:
            self.pooling = self.random_pooling
        elif self.pooling_module == FilterTypeEnum.Middle.value:
            self.pooling = self.middle_pooling
        elif self.pooling_module == FilterTypeEnum.Transformer.value:
            self.pooling_module = Encoder(512, 512, 1, 2)
            self.pooling = self.transformer_pooling
        else:
            raise NotImplementedError("No such method")

    def first_pooling(self, data):
        return data[0]

    def transformer_pooling(self, data):
        output, _ = self.pooling_module(data.unsqueeze(0), None)
        output = output.mean(dim=1).squeeze()
        return output

    def last_pooling(self, data):
        return data[-1]

    def random_pooling(self, data):
        return data[np.random.randint(len(data))]

    def middle_pooling(self, data):
        return data[int(len(data) / 2)]

    def max_pooling(self, data):
        return torch.max(data.unsqueeze(0), dim=1)[0].squeeze()

    def mean_pooling(self, data):
        return torch.mean(data.unsqueeze(0), dim=1).squeeze()

    def netvlad_pooling(self, data):
        return self.pooling_module(data.unsqueeze(0))

    def forward(self, src, similarities=None, similar_threshold=None):
        return self.main_forward(src, similarities, similar_threshold)

    def topk_similar(self, src, similarities=None):
        new_src = []
        with torch.no_grad():
            for n_sample, similarity_sample in enumerate(similarities):
                new_input = []
                indexes = torch.topk(similarity_sample, self.n_important_frames)[1]
                pre_index = -1
                indexes = indexes.sort()[0]
                for index in indexes:
                    data = src[n_sample][pre_index + 1:index + 1]
                    if len(data) > 1:
                        new_input.append(self.pooling(data))
                    else:
                        new_input.append(data.squeeze())
                    pre_index = index
                data = src[n_sample][pre_index + 1:]
                if len(data) > 1:
                    new_input.append(self.pooling(data))
                else:
                    new_input.append(data.squeeze())

                new_src.append(torch.stack(new_input))
        return new_src

    def threshold_similar(self, src, similarities=None, similar_threshold=None):
        if similar_threshold is None:
            similar_threshold = self.similar_threshold

        n_batch, n_frame, n_feature = src.shape
        new_src = []
        with torch.no_grad():
            for n_sample, similarity_sample in enumerate(similarities):
                similar_frames = []
                for i, similarity in enumerate(similarity_sample):
                    if similarity < similar_threshold:
                        similar_frames.append(i)

                new_input = []
                flag = False
                start = -1
                for i in range(n_frame):
                    if i in similar_frames:
                        flag = True
                    else:
                        if flag:
                            data = src[n_sample][start + 1:i + 1]
                            new_input.append(self.pooling(data))
                        else:
                            new_input.append(src[n_sample][i])
                        flag = False
                        start = i

                new_src.append(torch.stack(new_input))
        return new_src


class SpottingTransformer(nn.Module):
    """SimpleTransformer在10月27日修改"""
    def __init__(self, vocab_size, embedding_dim, n_layer, n_head, num_class, args):
        super().__init__()
        self.encoder_4_spot = SimpleTransformer(vocab_size, embedding_dim, n_layer, n_head,
                                                args.chunk_size * args.framerate - 1,
                                                dropout_rate=args.dropout_rate)
        self.similar_threshold = args.similar_threshold
        self.use_spot_loss = args.use_spot_loss
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, similarity, src_mask=None):
        output = self.encoder_4_spot(src, src_mask)
        return None, output

class SpottingTransformer2(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_layer, n_head, num_class, args):
        super().__init__()
        self.encoder_4_spot = Transformer0522(vocab_size, embedding_dim, n_layer, n_head,
                                                args.chunk_size * args.framerate - 1,
                                                dropout_rate=args.dropout_rate)
        self.similar_threshold = args.similar_threshold
        self.use_spot_loss = args.use_spot_loss
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, similarity, src_mask=None):
        output = self.encoder_4_spot(src, src_mask)
        return None, output


class SpottingTransformer3(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_layer, n_head, num_class, args):
        super().__init__()
        self.encoder_4_spot = Transformer0522_4_spotting(vocab_size, embedding_dim, n_layer, n_head,
                                                         args.chunk_size * args.framerate - 1,
                                                         dropout_rate=args.dropout_rate)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, similarity, src_mask=None):
        output = self.encoder_4_spot(src, src_mask)
        return None, output


class SpottingTransformer4(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_layer, n_head, num_class, args):
        super().__init__()
        self.encoder = Encoder(vocab_size, embedding_dim, n_layer, n_head)
        self.sigmoid = nn.Sigmoid()
        self.out = nn.Linear(embedding_dim, num_class + 1)
        self.dropout = nn.Dropout(p=args.dropout_rate)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, similarity, src_mask=None):
        output, enc_attn = self.encoder(src, src_mask)
        output = self.sigmoid(output)
        output = self.dropout(output)
        output = self.out(output)
        output = output.mean(dim=2)  # mean on frames in one chunk
        return None, output


class SpottingTransformer5(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_layer, n_head, num_class, args):
        super().__init__()
        self.encoder = Encoder(vocab_size, embedding_dim, n_layer, n_head)
        self.sigmoid = nn.Sigmoid()
        self.out = nn.Linear(embedding_dim, num_class + 1)
        self.dropout = nn.Dropout(p=args.dropout_rate)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, similarity, src_mask=None):
        output, enc_attn = self.encoder(src, src_mask)
        output = self.sigmoid(output)
        output = self.dropout(output)
        output = self.out(output)
        output = output.mean(dim=2)  # mean on frames in one chunk
        return None, output


class SpottingTransformer6(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_layer, n_head, num_class, args):
        super().__init__()
        if args.simplify_data:
            num_class = args.chunk_size - 1
        else:
            num_class = args.chunk_size * args.framerate - 1
        if args.add_flag_label:
            num_class += 1
        self.encoder_4_spot = Transformer0522(vocab_size, embedding_dim, n_layer, n_head,
                                              num_class,
                                              dropout_rate=args.dropout_rate)
        self.similar_threshold = args.similar_threshold
        self.use_spot_loss = args.use_spot_loss
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, similarity, src_mask=None):
        output = self.encoder_4_spot(src, src_mask)
        return None, output


class BestTransformer(nn.Module):

    def __init__(self, vocab_size, embedding_dim, N, heads, num_class, args):
        super().__init__()
        self.encoder_before = SimpleEncoder(vocab_size, embedding_dim, N, heads)
        self.encoder_after = SimpleEncoder(vocab_size, embedding_dim, N, heads)
        self.sigmoid = nn.Sigmoid()
        self.out = nn.Linear(embedding_dim, num_class + 1)
        self.dropout = nn.Dropout(p=args.dropout_rate)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_mask=None):
        batch_size, num_frame, num_feature = src.shape
        nb_frames_half = int(num_frame / 2)
        src_before = src[:, :nb_frames_half, :]
        src_after = src[:, nb_frames_half:, :]
        output_before, enc_attn_before = self.encoder_before(src_before, None)
        output_after, enc_attn_after = self.encoder_after(src_after, None)
        output = torch.cat((output_before, output_after), dim=1)

        output = output.mean(dim=1)  # mean on frames in one chunk
        output = self.sigmoid(output)
        output = self.dropout(output)
        output = self.out(output)
        # visual_attn(tuple(enc_attn_before))
        # batch, 4
        # return output, tuple(enc_attn_before), tuple(enc_attn_after)
        return output, None


class BestTransformer2(nn.Module):

    def __init__(self, vocab_size, embedding_dim, N, heads, num_class, args):
        super().__init__()
        self.encoder_before = SimpleEncoderNoPE(vocab_size, embedding_dim, N, heads)
        self.encoder_after = SimpleEncoderNoPE(vocab_size, embedding_dim, N, heads)
        self.sigmoid = nn.Sigmoid()
        self.out = nn.Linear(embedding_dim, num_class + 1)
        self.dropout = nn.Dropout(p=args.dropout_rate)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_mask=None):
        batch_size, num_frame, num_feature = src.shape
        nb_frames_half = int(num_frame / 2)
        src_before = src[:, :nb_frames_half, :]
        src_after = src[:, nb_frames_half:, :]
        output_before, enc_attn_before = self.encoder_before(src_before, None)
        output_after, enc_attn_after = self.encoder_after(src_after, None)
        output = torch.cat((output_before, output_after), dim=1)

        output = output.mean(dim=1)  # mean on frames in one chunk
        output = self.sigmoid(output)
        output = self.dropout(output)
        output = self.out(output)
        return output, None


class BestTransformer3(nn.Module):

    def __init__(self, vocab_size, embedding_dim, N, heads, num_class, args):
        super().__init__()
        self.encoder_before = MixEncoder(vocab_size, embedding_dim, N, heads)
        self.encoder_after = MixEncoder(vocab_size, embedding_dim, N, heads)
        self.sigmoid = nn.Sigmoid()
        self.out = nn.Linear(embedding_dim, num_class + 1)
        self.dropout = nn.Dropout(p=args.dropout_rate)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_mask=None):
        batch_size, num_frame, num_feature = src.shape
        nb_frames_half = int(num_frame / 2)
        src_before = src[:, :nb_frames_half, :]
        src_after = src[:, nb_frames_half:, :]
        output_before, enc_attn_before = self.encoder_before(src_before, None)
        output_after, enc_attn_after = self.encoder_after(src_after, None)
        output = torch.cat((output_before, output_after), dim=1)

        output = output.mean(dim=1)  # mean on frames in one chunk
        output = self.sigmoid(output)
        output = self.dropout(output)
        output = self.out(output)
        return output, None

class BestTransformer4(nn.Module):

    def __init__(self, vocab_size, embedding_dim, N, heads, num_class, args):
        super().__init__()
        self.encoder_before = MixEncoderNoPE(vocab_size, embedding_dim, N, heads)
        self.encoder_after = MixEncoderNoPE(vocab_size, embedding_dim, N, heads)
        self.sigmoid = nn.Sigmoid()
        self.out = nn.Linear(embedding_dim, num_class + 1)
        self.dropout = nn.Dropout(p=args.dropout_rate)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_mask=None):
        batch_size, num_frame, num_feature = src.shape
        nb_frames_half = int(num_frame / 2)
        src_before = src[:, :nb_frames_half, :]
        src_after = src[:, nb_frames_half:, :]
        output_before, enc_attn_before = self.encoder_before(src_before, None)
        output_after, enc_attn_after = self.encoder_after(src_after, None)
        output = torch.cat((output_before, output_after), dim=1)

        output = output.mean(dim=1)  # mean on frames in one chunk
        output = self.sigmoid(output)
        output = self.dropout(output)
        output = self.out(output)
        return output, None


class BestTransformer5(nn.Module):

    def __init__(self, vocab_size, embedding_dim, N, heads, num_class, args):
        super().__init__()
        self.encoder_before = PEwoPEMixEncoder(vocab_size, embedding_dim, N, heads)
        self.encoder_after = PEwoPEMixEncoder(vocab_size, embedding_dim, N, heads)
        self.sigmoid = nn.Sigmoid()
        self.out = nn.Linear(embedding_dim, num_class + 1)
        self.dropout = nn.Dropout(p=0.1)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_mask=None):
        batch_size, num_frame, num_feature = src.shape
        nb_frames_half = int(num_frame / 2)
        src_before = src[:, :nb_frames_half, :]
        src_after = src[:, nb_frames_half:, :]
        output_before, enc_attn_before = self.encoder_before(src_before, None)
        output_after, enc_attn_after = self.encoder_after(src_after, None)
        output = torch.cat((output_before, output_after), dim=1)

        output = output.mean(dim=1)  # mean on frames in one chunk
        output = self.sigmoid(output)
        output = self.dropout(output)
        output = self.out(output)
        return output, None


class BestTransformer6(nn.Module):

    def __init__(self, vocab_size, embedding_dim, N, heads, num_class, args):
        super().__init__()
        self.encoder_before = PEwoPEEncoder(vocab_size, embedding_dim, N, heads)
        self.encoder_after = PEwoPEEncoder(vocab_size, embedding_dim, N, heads)
        self.sigmoid = nn.Sigmoid()
        self.out = nn.Linear(embedding_dim, num_class + 1)
        self.dropout = nn.Dropout(p=0.1)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_mask=None):
        batch_size, num_frame, num_feature = src.shape
        nb_frames_half = int(num_frame / 2)
        src_before = src[:, :nb_frames_half, :]
        src_after = src[:, nb_frames_half:, :]
        output_before, enc_attn_before = self.encoder_before(src_before, None)
        output_after, enc_attn_after = self.encoder_after(src_after, None)
        output = torch.cat((output_before, output_after), dim=1)

        output = output.mean(dim=1)  # mean on frames in one chunk
        output = self.sigmoid(output)
        output = self.dropout(output)
        output = self.out(output)
        return output, None


class BestTransformer7(nn.Module):

    def __init__(self, vocab_size, embedding_dim, N, heads, num_class, args):
        super().__init__()
        self.encoder_before = PEwoPESimpleEncoder(vocab_size, embedding_dim, N, heads)
        self.encoder_after = PEwoPESimpleEncoder(vocab_size, embedding_dim, N, heads)
        self.sigmoid = nn.Sigmoid()
        self.out = nn.Linear(embedding_dim, num_class + 1)
        self.dropout = nn.Dropout(p=args.dropout_rate)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_mask=None):
        batch_size, num_frame, num_feature = src.shape
        nb_frames_half = int(num_frame / 2)
        src_before = src[:, :nb_frames_half, :]
        src_after = src[:, nb_frames_half:, :]
        output_before, enc_attn_before = self.encoder_before(src_before, None)
        output_after, enc_attn_after = self.encoder_after(src_after, None)
        output = torch.cat((output_before, output_after), dim=1)

        output = output.mean(dim=1)  # mean on frames in one chunk
        output = self.sigmoid(output)
        output = self.dropout(output)
        output = self.out(output)
        return output, None


class BestTransformer8(nn.Module):

    def __init__(self, vocab_size, embedding_dim, N, heads, num_class, args, max_seq_len=200):
        super().__init__()
        self.encoder_before = SimpleEncoder(vocab_size, embedding_dim, N, heads, max_seq_len=max_seq_len // 2,
                                            pe_type="pre")
        self.encoder_after = SimpleEncoder(vocab_size, embedding_dim, N, heads, max_seq_len=max_seq_len // 2,
                                           pe_type="post")
        self.sigmoid = nn.Sigmoid()
        self.out = nn.Linear(embedding_dim, num_class + 1)
        self.dropout = nn.Dropout(p=args.dropout_rate)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_mask=None):
        batch_size, num_frame, num_feature = src.shape
        nb_frames_half = int(num_frame / 2)
        src_before = src[:, :nb_frames_half, :]
        src_after = src[:, nb_frames_half:, :]
        output_before, enc_attn_before = self.encoder_before(src_before, None)
        output_after, enc_attn_after = self.encoder_after(src_after, None)
        output = torch.cat((output_before, output_after), dim=1)

        output = output.mean(dim=1)  # mean on frames in one chunk
        output = self.sigmoid(output)
        output = self.dropout(output)
        output = self.out(output)
        # visual_attn(tuple(enc_attn_before))
        # batch, 4
        # return output, tuple(enc_attn_before), tuple(enc_attn_after)
        return output, None


class BestTransformer9(nn.Module):

    def __init__(self, vocab_size, embedding_dim, N, heads, num_class, args, max_seq_len=200):
        super().__init__()
        self.encoder_before = SimpleEncoder(vocab_size, embedding_dim, N, heads, max_seq_len=max_seq_len)
        self.encoder_after = SimpleEncoder(vocab_size, embedding_dim, N, heads, max_seq_len=max_seq_len)
        self.sigmoid = nn.Sigmoid()
        self.out = nn.Linear(embedding_dim, num_class + 1)
        self.dropout = nn.Dropout(p=args.dropout_rate)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_mask=None):
        batch_size, num_frame, num_feature = src.shape
        nb_frames_half = int(num_frame / 2)
        src_before = src[:, nb_frames_half:, :]
        src_after = src[:, :nb_frames_half, :]
        output_before, enc_attn_before = self.encoder_before(src_before, None)
        output_after, enc_attn_after = self.encoder_after(src_after, None)
        output = torch.cat((output_before, output_after), dim=1)

        output = output.mean(dim=1)  # mean on frames in one chunk
        output = self.sigmoid(output)
        output = self.dropout(output)
        output = self.out(output)
        # visual_attn(tuple(enc_attn_before))
        # batch, 4
        # return output, tuple(enc_attn_before), tuple(enc_attn_after)
        return output, None


class BestTransformer10(nn.Module):
    def __init__(self, vocab_size, embedding_dim, N, heads, num_class, args, max_seq_len=200):
        super().__init__()
        self.encoder_1 = SimpleEncoder(vocab_size, embedding_dim, N, heads, max_seq_len=max_seq_len)
        self.encoder_2 = SimpleEncoder(vocab_size, embedding_dim, N, heads, max_seq_len=max_seq_len)
        self.encoder_3 = SimpleEncoder(vocab_size, embedding_dim, N, heads, max_seq_len=max_seq_len)
        self.sigmoid = nn.Sigmoid()
        self.out = nn.Linear(embedding_dim, num_class + 1)
        self.dropout = nn.Dropout(p=args.dropout_rate)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_mask=None):
        batch_size, num_frame, num_feature = src.shape
        nb_frames_half = int(num_frame / 3)
        src_1 = src[:, :nb_frames_half, :]
        src_2 = src[:, nb_frames_half:-nb_frames_half, :]
        src_3 = src[:, -nb_frames_half:, :]
        output_1, _ = self.encoder_1(src_1, None)
        output_2, _ = self.encoder_2(src_2, None)
        output_3, _ = self.encoder_3(src_3, None)
        output = torch.cat((output_1, output_2, output_3), dim=1)

        output = output.mean(dim=1)  # mean on frames in one chunk
        output = self.sigmoid(output)
        output = self.dropout(output)
        output = self.out(output)
        return output, None


class BestTransformer11(nn.Module):
    def __init__(self, vocab_size, embedding_dim, N, heads, num_class, args, max_seq_len=200):
        super().__init__()
        self.encoder_1 = SimpleEncoder(vocab_size, embedding_dim, N, heads, max_seq_len=max_seq_len)
        self.encoder_2 = SimpleEncoder(vocab_size, embedding_dim, N, heads, max_seq_len=max_seq_len)
        self.encoder_3 = SimpleEncoder(vocab_size, embedding_dim, N, heads, max_seq_len=max_seq_len)
        self.encoder_4 = SimpleEncoder(vocab_size, embedding_dim, N, heads, max_seq_len=max_seq_len)
        self.sigmoid = nn.Sigmoid()
        self.out = nn.Linear(embedding_dim, num_class + 1)
        self.dropout = nn.Dropout(p=args.dropout_rate)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_mask=None):
        batch_size, num_frame, num_feature = src.shape
        nb_frames_half = int(num_frame / 4)
        src_1 = src[:, :nb_frames_half, :]
        src_2 = src[:, nb_frames_half:2 * nb_frames_half, :]
        src_3 = src[:, 2 * nb_frames_half:-nb_frames_half, :]
        src_4 = src[:, -nb_frames_half:, :]
        output_1, _ = self.encoder_1(src_1, None)
        output_2, _ = self.encoder_2(src_2, None)
        output_3, _ = self.encoder_3(src_3, None)
        output_4, _ = self.encoder_4(src_4, None)
        output = torch.cat((output_1, output_2, output_3, output_4), dim=1)

        output = output.mean(dim=1)  # mean on frames in one chunk
        output = self.sigmoid(output)
        output = self.dropout(output)
        output = self.out(output)
        return output, None


class BestTransformer12(nn.Module):
    def __init__(self, vocab_size, embedding_dim, N, heads, num_class, args, max_seq_len=200):
        super().__init__()
        self.encoder_1 = SimpleEncoder(vocab_size, embedding_dim, N, heads, max_seq_len=max_seq_len)
        self.encoder_2 = SimpleEncoder(vocab_size, embedding_dim, N, heads, max_seq_len=max_seq_len)
        self.encoder_3 = SimpleEncoder(vocab_size, embedding_dim, N, heads, max_seq_len=max_seq_len)
        self.sigmoid = nn.Sigmoid()
        self.out = nn.Linear(embedding_dim, num_class + 1)
        self.dropout = nn.Dropout(p=args.dropout_rate)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_mask=None):
        batch_size, num_frame, num_feature = src.shape
        nb_frames_half = int(num_frame / 3)
        src_1 = src[:, :nb_frames_half, :]
        src_2 = src[:, nb_frames_half:-nb_frames_half, :]
        src_3 = src[:, -nb_frames_half:, :]
        output_1, _ = self.encoder_1(src_1, None)
        output_2, _ = self.encoder_2(src_2, None)
        output_3, _ = self.encoder_3(src_3, None)
        output = torch.cat((output_3, output_2, output_1), dim=1)

        output = output.mean(dim=1)  # mean on frames in one chunk
        output = self.sigmoid(output)
        output = self.dropout(output)
        output = self.out(output)
        return output, None


class BestTransformer13(nn.Module):
    def __init__(self, vocab_size, embedding_dim, N, heads, num_class, args, max_seq_len=200):
        super().__init__()
        self.encoder_1 = SimpleEncoder(vocab_size, embedding_dim, N, heads, max_seq_len=max_seq_len)
        self.encoder_2 = SimpleEncoder(vocab_size, embedding_dim, N, heads, max_seq_len=max_seq_len)
        self.encoder_3 = SimpleEncoder(vocab_size, embedding_dim, N, heads, max_seq_len=max_seq_len)
        self.encoder_4 = SimpleEncoder(vocab_size, embedding_dim, N, heads, max_seq_len=max_seq_len)
        self.sigmoid = nn.Sigmoid()
        self.out = nn.Linear(embedding_dim, num_class + 1)
        self.dropout = nn.Dropout(p=args.dropout_rate)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_mask=None):
        batch_size, num_frame, num_feature = src.shape
        nb_frames_half = int(num_frame / 4)
        src_1 = src[:, :nb_frames_half, :]
        src_2 = src[:, nb_frames_half:2 * nb_frames_half, :]
        src_3 = src[:, 2 * nb_frames_half:-nb_frames_half, :]
        src_4 = src[:, -nb_frames_half:, :]
        output_1, _ = self.encoder_1(src_1, None)
        output_2, _ = self.encoder_2(src_2, None)
        output_3, _ = self.encoder_3(src_3, None)
        output_4, _ = self.encoder_4(src_4, None)
        output = torch.cat((output_4, output_3, output_2, output_1), dim=1)

        output = output.mean(dim=1)  # mean on frames in one chunk
        output = self.sigmoid(output)
        output = self.dropout(output)
        output = self.out(output)
        return output, None


class BestTransformer14(nn.Module):

    def __init__(self, vocab_size, embedding_dim, N, heads, num_class, args, max_seq_len=200):
        super().__init__()
        self.encoder_before = SimpleEncoder(vocab_size, embedding_dim, N, heads, max_seq_len=max_seq_len)
        self.encoder_after = SimpleEncoder(vocab_size, embedding_dim, N, heads, max_seq_len=max_seq_len)
        self.sigmoid = nn.Sigmoid()
        self.out = nn.Linear(embedding_dim, num_class + 1)
        self.dropout = nn.Dropout(p=args.dropout_rate)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_mask=None):
        batch_size, num_frame, num_feature = src.shape
        nb_frames_half = int(num_frame / 2)
        output_before, enc_attn_before = self.encoder_before(src, None)
        output_after, enc_attn_after = self.encoder_after(src, None)
        output = torch.cat((output_before[:, :nb_frames_half, :], output_after[:, :nb_frames_half:, :]), dim=1)

        output = output.mean(dim=1)  # mean on frames in one chunk
        output = self.sigmoid(output)
        output = self.dropout(output)
        output = self.out(output)
        # visual_attn(tuple(enc_attn_before))
        # batch, 4
        # return output, tuple(enc_attn_before), tuple(enc_attn_after)
        return output, None


class BestTransformer15(nn.Module):

    def __init__(self, vocab_size, embedding_dim, N, heads, num_class, args, max_seq_len=200):
        super().__init__()
        self.filter = nn.Linear(2048, 512)
        self.encoder_before = SimpleEncoder(vocab_size, embedding_dim, N, heads, max_seq_len=max_seq_len)
        self.encoder_after = SimpleEncoder(vocab_size, embedding_dim, N, heads, max_seq_len=max_seq_len)
        self.sigmoid = nn.Sigmoid()
        self.out = nn.Linear(embedding_dim, num_class + 1)
        self.dropout = nn.Dropout(p=args.dropout_rate)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_mask=None):
        batch_size, num_frame, num_feature = src.shape
        src = self.filter(src)
        nb_frames_half = int(num_frame / 2)
        src_before = src[:, nb_frames_half:, :]
        src_after = src[:, :nb_frames_half, :]
        output_before, enc_attn_before = self.encoder_before(src_before, None)
        output_after, enc_attn_after = self.encoder_after(src_after, None)
        output = torch.cat((output_before, output_after), dim=1)

        output = output.mean(dim=1)  # mean on frames in one chunk
        output = self.sigmoid(output)
        output = self.dropout(output)
        output = self.out(output)
        # visual_attn(tuple(enc_attn_before))
        # batch, 4
        # return output, tuple(enc_attn_before), tuple(enc_attn_after)
        return output, None


class BestTransformer16(nn.Module):

    def __init__(self, vocab_size, embedding_dim, N, heads, num_class, args, max_seq_len=200):
        super().__init__()
        self.filter = nn.Linear(2048, 512)
        self.similar_filter_before = SimpleEncoderNoPE(vocab_size, embedding_dim, 1, 1)
        self.similar_filter_after = SimpleEncoderNoPE(vocab_size, embedding_dim, 1, 1)
        self.encoder_before = SimpleEncoder(embedding_dim, embedding_dim, N, heads, max_seq_len=max_seq_len)
        self.encoder_after = SimpleEncoder(embedding_dim, embedding_dim, N, heads, max_seq_len=max_seq_len)
        self.sigmoid = nn.Sigmoid()
        self.out = nn.Linear(embedding_dim, num_class + 1)
        # self.dropout = nn.Dropout(p=args.dropout_rate)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_mask=None):
        batch_size, num_frame, num_feature = src.shape
        src = self.filter(src)
        nb_frames_half = int(num_frame / 2)
        src_before = src[:, :nb_frames_half, :]
        src_after = src[:, nb_frames_half:, :]
        src_before, _ = self.similar_filter_before(src_before, None)
        src_after, _ = self.similar_filter_after(src_after, None)
        output_before, enc_attn_before = self.encoder_before(src_before, None)
        output_after, enc_attn_after = self.encoder_after(src_after, None)
        output = torch.cat((output_before, output_after), dim=1)

        output = output.mean(dim=1)  # mean on frames in one chunk
        output = self.sigmoid(output)
        # output = self.dropout(output)
        output = self.out(output)
        # visual_attn(tuple(enc_attn_before))
        # batch, 4
        # return output, tuple(enc_attn_before), tuple(enc_attn_after)
        return output, None

class BestTransformer17(nn.Module):

    def __init__(self, vocab_size, embedding_dim, N, heads, num_class, args, max_seq_len=200):
        super().__init__()
        self.filter = nn.Linear(2048, 512)
        self.similar_filter_before = SimpleEncoderNoPE(vocab_size, embedding_dim, 1, 1)
        self.similar_filter_after = SimpleEncoderNoPE(vocab_size, embedding_dim, 1, 1)
        self.encoder_before = SimpleEncoder(vocab_size, embedding_dim, N, heads, max_seq_len=max_seq_len)
        self.encoder_after = SimpleEncoder(vocab_size, embedding_dim, N, heads, max_seq_len=max_seq_len)
        self.sigmoid = nn.Sigmoid()
        self.out = nn.Linear(embedding_dim, num_class + 1)
        # self.dropout = nn.Dropout(p=args.dropout_rate)

        # for p in self.parameters():
        #     if p.dim() > 1:
        #         nn.init.xavier_uniform_(p)

    def forward(self, src, src_mask=None):
        batch_size, num_frame, num_feature = src.shape
        src = self.filter(src)
        nb_frames_half = int(num_frame / 2)
        src_before = src[:, :nb_frames_half, :]
        src_after = src[:, nb_frames_half:, :]
        src_before, _ = self.similar_filter_before(src_before, None)
        src_after, _ = self.similar_filter_after(src_after, None)
        output_before, enc_attn_before = self.encoder_before(src_before, None)
        output_after, enc_attn_after = self.encoder_after(src_after, None)
        output = torch.cat((output_before, output_after), dim=1)

        output = output.mean(dim=1)  # mean on frames in one chunk
        output = self.sigmoid(output)
        # output = self.dropout(output)
        output = self.out(output)
        # visual_attn(tuple(enc_attn_before))
        # batch, 4
        # return output, tuple(enc_attn_before), tuple(enc_attn_after)
        return output, None


class BestTransformer18(nn.Module):
    # 用来学习百度的特征量

    def __init__(self, vocab_size, embedding_dim, N, heads, num_class, args, max_seq_len=200):
        super().__init__()
        self.filter = nn.Linear(8576, 512)
        self.similar_filter_before = SimpleEncoderNoPE(vocab_size, embedding_dim, 1, 1)
        self.similar_filter_after = SimpleEncoderNoPE(vocab_size, embedding_dim, 1, 1)
        self.encoder_before = SimpleEncoder(vocab_size, embedding_dim, N, heads, max_seq_len=max_seq_len)
        self.encoder_after = SimpleEncoder(vocab_size, embedding_dim, N, heads, max_seq_len=max_seq_len)
        self.sigmoid = nn.Sigmoid()
        self.out = nn.Linear(embedding_dim, num_class + 1)
        # self.dropout = nn.Dropout(p=args.dropout_rate)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_mask=None):
        batch_size, num_frame, num_feature = src.shape
        src = self.filter(src)
        nb_frames_half = int(num_frame / 2)
        src_before = src[:, :nb_frames_half, :]
        src_after = src[:, nb_frames_half:, :]
        src_before, _ = self.similar_filter_before(src_before, None)
        src_after, _ = self.similar_filter_after(src_after, None)
        output_before, enc_attn_before = self.encoder_before(src_before, None)
        output_after, enc_attn_after = self.encoder_after(src_after, None)
        output = torch.cat((output_before, output_after), dim=1)

        output = output.mean(dim=1)  # mean on frames in one chunk
        output = self.sigmoid(output)
        # output = self.dropout(output)
        output = self.out(output)
        # visual_attn(tuple(enc_attn_before))
        # batch, 4
        # return output, tuple(enc_attn_before), tuple(enc_attn_after)
        return output, None



class BestTransformer19(nn.Module):

    def __init__(self, vocab_size, embedding_dim, N, heads, num_class, args, max_seq_len=200):
        super().__init__()
        # self.filter = nn.Linear(2048, 512)
        self.similar_filter_before = SimpleEncoderNoPE(vocab_size, embedding_dim, 1, 1)
        self.similar_filter_after = SimpleEncoderNoPE(vocab_size, embedding_dim, 1, 1)
        self.encoder_before = SimpleEncoder(embedding_dim, embedding_dim, N, heads, max_seq_len=max_seq_len)
        self.encoder_after = SimpleEncoder(embedding_dim, embedding_dim, N, heads, max_seq_len=max_seq_len)
        self.sigmoid = nn.Sigmoid()
        self.out = nn.Linear(embedding_dim, num_class + 1)
        # self.dropout = nn.Dropout(p=args.dropout_rate)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_mask=None):
        batch_size, num_frame, num_feature = src.shape
        # src = self.filter(src)
        nb_frames_half = int(num_frame / 2)
        src_before = src[:, :nb_frames_half, :]
        src_after = src[:, nb_frames_half:, :]
        src_before, _ = self.similar_filter_before(src_before, None)
        src_after, _ = self.similar_filter_after(src_after, None)
        output_before, enc_attn_before = self.encoder_before(src_before, None)
        output_after, enc_attn_after = self.encoder_after(src_after, None)
        output = torch.cat((output_before, output_after), dim=1)

        output = output.mean(dim=1)  # mean on frames in one chunk
        output = self.sigmoid(output)
        # output = self.dropout(output)
        output = self.out(output)
        # visual_attn(tuple(enc_attn_before))
        # batch, 4
        # return output, tuple(enc_attn_before), tuple(enc_attn_after)
        return output, None


class BestTransformer20(nn.Module):
    # 基于16，修正了multi-attent的bug
    def __init__(self, vocab_size, embedding_dim, N, heads, num_class, args, max_seq_len=200):
        super().__init__()
        self.filter = nn.Linear(2048, 512)
        self.similar_filter_before = SimpleEncoderNoPE(vocab_size, embedding_dim, 1, 1)
        self.similar_filter_after = SimpleEncoderNoPE(vocab_size, embedding_dim, 1, 1)
        self.encoder_before = SimpleEncoderTrue(embedding_dim, embedding_dim, N, heads, max_seq_len=max_seq_len)
        self.encoder_after = SimpleEncoderTrue(embedding_dim, embedding_dim, N, heads, max_seq_len=max_seq_len)
        self.sigmoid = nn.Sigmoid()
        self.out = nn.Linear(embedding_dim, num_class + 1)
        # self.dropout = nn.Dropout(p=args.dropout_rate)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_mask=None):
        batch_size, num_frame, num_feature = src.shape
        src = self.filter(src)
        nb_frames_half = int(num_frame / 2)
        src_before = src[:, :nb_frames_half, :]
        src_after = src[:, nb_frames_half:, :]
        src_before, _ = self.similar_filter_before(src_before, None)
        src_after, _ = self.similar_filter_after(src_after, None)
        output_before, enc_attn_before = self.encoder_before(src_before, None)
        output_after, enc_attn_after = self.encoder_after(src_after, None)
        output = torch.cat((output_before, output_after), dim=1)

        output = output.mean(dim=1)  # mean on frames in one chunk
        output = self.sigmoid(output)
        # output = self.dropout(output)
        output = self.out(output)
        # visual_attn(tuple(enc_attn_before))
        # batch, 4
        # return output, tuple(enc_attn_before), tuple(enc_attn_after)
        return output, None




class BestTransformer21(nn.Module):

    def __init__(self, vocab_size, embedding_dim, N, heads, num_class, args, max_seq_len=200):
        super().__init__()
        # self.filter = nn.Linear(2048, 512)
        self.similar_filter_before = SimpleEncoderNoPE(vocab_size, embedding_dim, 1, 1)
        self.similar_filter_after = SimpleEncoderNoPE(vocab_size, embedding_dim, 1, 1)
        self.encoder_before = SimpleEncoderTrue(embedding_dim, embedding_dim, N, heads, max_seq_len=max_seq_len)
        self.encoder_after = SimpleEncoderTrue(embedding_dim, embedding_dim, N, heads, max_seq_len=max_seq_len)
        self.sigmoid = nn.Sigmoid()
        self.out = nn.Linear(embedding_dim, num_class + 1)
        # self.dropout = nn.Dropout(p=args.dropout_rate)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_mask=None):
        batch_size, num_frame, num_feature = src.shape
        # src = self.filter(src)
        nb_frames_half = int(num_frame / 2)
        src_before = src[:, :nb_frames_half, :]
        src_after = src[:, nb_frames_half:, :]
        src_before, _ = self.similar_filter_before(src_before, None)
        src_after, _ = self.similar_filter_after(src_after, None)
        output_before, enc_attn_before = self.encoder_before(src_before, None)
        output_after, enc_attn_after = self.encoder_after(src_after, None)
        output = torch.cat((output_before, output_after), dim=1)

        output = output.mean(dim=1)  # mean on frames in one chunk
        output = self.sigmoid(output)
        # output = self.dropout(output)
        output = self.out(output)
        # visual_attn(tuple(enc_attn_before))
        # batch, 4
        # return output, tuple(enc_attn_before), tuple(enc_attn_after)
        return output, None


class BestTransformer20_4_atten(nn.Module):
    # 基于16，修正了multi-attent的bug
    def __init__(self, vocab_size, embedding_dim, N, heads, num_class, args, max_seq_len=200):
        super().__init__()
        self.filter = nn.Linear(2048, 512)
        self.similar_filter_before = SimpleEncoderNoPE(vocab_size, embedding_dim, 1, 1)
        self.similar_filter_after = SimpleEncoderNoPE(vocab_size, embedding_dim, 1, 1)
        self.encoder_before = SimpleEncoderTrue(embedding_dim, embedding_dim, N, heads, max_seq_len=max_seq_len)
        self.encoder_after = SimpleEncoderTrue(embedding_dim, embedding_dim, N, heads, max_seq_len=max_seq_len)
        self.sigmoid = nn.Sigmoid()
        self.out = nn.Linear(embedding_dim, num_class + 1)
        # self.dropout = nn.Dropout(p=args.dropout_rate)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_mask=None):
        batch_size, num_frame, num_feature = src.shape
        # src = self.filter(src)
        nb_frames_half = int(num_frame / 2)
        src_before = src[:, :nb_frames_half, :]
        src_after = src[:, nb_frames_half:, :]
        src_before, _ = self.similar_filter_before(src_before, None)
        src_after, _ = self.similar_filter_after(src_after, None)
        output_before, enc_attn_before = self.encoder_before(src_before, None)
        output_after, enc_attn_after = self.encoder_after(src_after, None)

        attn_score_before = visual_attn(enc_attn_before, n_layer=1)
        attn_score_after = visual_attn(enc_attn_after, n_layer=1)
        attn = torch.cat((attn_score_before, attn_score_after), dim=1)

        output = torch.cat((output_before, output_after), dim=1)

        output = output.mean(dim=1)  # mean on frames in one chunk
        output = self.sigmoid(output)
        output = self.out(output)
        return output, attn


class BestTransformer22(nn.Module):

    def __init__(self, vocab_size, embedding_dim, N, heads, num_class, args, max_seq_len=200):
        super().__init__()
        self.similar_filter_before = SimpleEncoderNoPETrue(vocab_size, embedding_dim, N, heads)
        self.similar_filter_after = SimpleEncoderNoPETrue(vocab_size, embedding_dim, N, heads)
        self.sigmoid = nn.Sigmoid()
        self.out = nn.Linear(embedding_dim, num_class + 1)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_mask=None):
        batch_size, num_frame, num_feature = src.shape
        # src = self.filter(src)
        nb_frames_half = int(num_frame / 2)
        src_before = src[:, :nb_frames_half, :]
        src_after = src[:, nb_frames_half:, :]
        output_before, _ = self.similar_filter_before(src_before, None)
        output_after, _ = self.similar_filter_after(src_after, None)
        output = torch.cat((output_before, output_after), dim=1)

        output = output.mean(dim=1)  # mean on frames in one chunk
        output = self.sigmoid(output)
        output = self.out(output)
        return output, None


class BestTransformer23(nn.Module):

    def __init__(self, vocab_size, embedding_dim, N, heads, num_class, args, max_seq_len=200):
        super().__init__()
        self.similar_filter_before = SimpleEncoderTrue(vocab_size, embedding_dim, N, heads)
        self.similar_filter_after = SimpleEncoderTrue(vocab_size, embedding_dim, N, heads)
        self.sigmoid = nn.Sigmoid()
        self.out = nn.Linear(embedding_dim, num_class + 1)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_mask=None):
        batch_size, num_frame, num_feature = src.shape
        # src = self.filter(src)
        nb_frames_half = int(num_frame / 2)
        src_before = src[:, :nb_frames_half, :]
        src_after = src[:, nb_frames_half:, :]
        output_before, _ = self.similar_filter_before(src_before, None)
        output_after, _ = self.similar_filter_after(src_after, None)
        output = torch.cat((output_before, output_after), dim=1)

        output = output.mean(dim=1)  # mean on frames in one chunk
        output = self.sigmoid(output)
        output = self.out(output)
        return output, None


class BestTransformer24(nn.Module):
    "Aad a linear based on BestTransformer22"

    def __init__(self, vocab_size, embedding_dim, N, heads, num_class, args, max_seq_len=200):
        super().__init__()
        self.linear1 = nn.Linear(vocab_size, 512)
        self.similar_filter_before = SimpleEncoderNoPETrue(512, embedding_dim, N, heads)
        self.similar_filter_after = SimpleEncoderNoPETrue(512, embedding_dim, N, heads)
        self.sigmoid = nn.Sigmoid()
        self.out = nn.Linear(embedding_dim, num_class + 1)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_mask=None):
        batch_size, num_frame, num_feature = src.shape
        src = self.linear1(src)
        # src = self.filter(src)
        nb_frames_half = int(num_frame / 2)
        src_before = src[:, :nb_frames_half, :]
        src_after = src[:, nb_frames_half:, :]
        output_before, _ = self.similar_filter_before(src_before, None)
        output_after, _ = self.similar_filter_after(src_after, None)
        output = torch.cat((output_before, output_after), dim=1)

        output = output.mean(dim=1)  # mean on frames in one chunk
        output = self.sigmoid(output)
        output = self.out(output)
        return output, None


class FIE(nn.Module):

    def __init__(self, vocab_size, embedding_dim, N, heads, num_class, args, max_seq_len=200):
        super().__init__()
        # if vocab_size != embedding_dim:
        self.linear1 = nn.Linear(vocab_size, embedding_dim)
        self.similar_filter_before = SimpleEncoderNoPETrue(embedding_dim, embedding_dim, N, heads)
        self.similar_filter_after = SimpleEncoderNoPETrue(embedding_dim, embedding_dim, N, heads)
        self.sigmoid = nn.Sigmoid()
        self.out = nn.Linear(embedding_dim, num_class + 1)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_mask=None):
        batch_size, num_frame, num_feature = src.shape
        src = self.linear1(src)
        nb_frames_half = int(num_frame / 2)
        src_before = src[:, :nb_frames_half, :]
        src_after = src[:, nb_frames_half:, :]
        output_before, _ = self.similar_filter_before(src_before, None)
        output_after, _ = self.similar_filter_after(src_after, None)
        output = torch.cat((output_before, output_after), dim=1)

        output = output.mean(dim=1)
        output = self.sigmoid(output)
        output = self.out(output)
        return output, None


class FIEOneEncoder(nn.Module):

    def __init__(self, vocab_size, embedding_dim, N, heads, num_class, args, max_seq_len=200):
        super().__init__()
        self.linear1 = nn.Linear(vocab_size, embedding_dim)
        self.fie = SimpleEncoderNoPETrue(embedding_dim, embedding_dim, N, heads)
        self.sigmoid = nn.Sigmoid()
        self.out = nn.Linear(embedding_dim, num_class + 1)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_mask=None):
        batch_size, num_frame, num_feature = src.shape
        src = self.linear1(src)
        output, _ = self.fie(src, None)

        output = output.mean(dim=1)
        output = self.sigmoid(output)
        output = self.out(output)
        return output, None


class FIEThreeEncoder(nn.Module):

    def __init__(self, vocab_size, embedding_dim, N, heads, num_class, args, max_seq_len=200):
        super().__init__()
        self.linear1 = nn.Linear(vocab_size, embedding_dim)
        self.fie_1 = SimpleEncoderNoPETrue(embedding_dim, embedding_dim, N, heads)
        self.fie_2 = SimpleEncoderNoPETrue(embedding_dim, embedding_dim, N, heads)
        self.fie_3 = SimpleEncoderNoPETrue(embedding_dim, embedding_dim, N, heads)
        self.sigmoid = nn.Sigmoid()
        self.out = nn.Linear(embedding_dim, num_class + 1)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_mask=None):
        batch_size, num_frame, num_feature = src.shape
        src = self.linear1(src)
        nb_frames_half = int(num_frame / 3)
        src_1 = src[:, :nb_frames_half, :]
        src_2 = src[:, nb_frames_half:-nb_frames_half, :]
        src_3 = src[:, -nb_frames_half:, :]
        output_1, _ = self.fie_1(src_1, None)
        output_2, _ = self.fie_2(src_2, None)
        output_3, _ = self.fie_3(src_3, None)
        output = torch.cat((output_1, output_2, output_3), dim=1)

        output = output.mean(dim=1)
        output = self.sigmoid(output)
        output = self.out(output)
        return output, None


class FIEFourEncoder(nn.Module):

    def __init__(self, vocab_size, embedding_dim, N, heads, num_class, args, max_seq_len=200):
        super().__init__()
        self.linear1 = nn.Linear(vocab_size, embedding_dim)
        self.fie_1 = SimpleEncoderNoPETrue(embedding_dim, embedding_dim, N, heads)
        self.fie_2 = SimpleEncoderNoPETrue(embedding_dim, embedding_dim, N, heads)
        self.fie_3 = SimpleEncoderNoPETrue(embedding_dim, embedding_dim, N, heads)
        self.fie_4 = SimpleEncoderNoPETrue(embedding_dim, embedding_dim, N, heads)
        self.sigmoid = nn.Sigmoid()
        self.out = nn.Linear(embedding_dim, num_class + 1)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_mask=None):
        batch_size, num_frame, num_feature = src.shape
        src = self.linear1(src)
        nb_frames_half = int(num_frame / 4)
        src_1 = src[:, :nb_frames_half, :]
        src_2 = src[:, nb_frames_half:2 * nb_frames_half, :]
        src_3 = src[:, 2 * nb_frames_half:-nb_frames_half, :]
        src_4 = src[:, -nb_frames_half:, :]
        output_1, _ = self.fie_1(src_1, None)
        output_2, _ = self.fie_2(src_2, None)
        output_3, _ = self.fie_3(src_3, None)
        output_4, _ = self.fie_4(src_4, None)
        output = torch.cat((output_1, output_2, output_3, output_4), dim=1)

        output = output.mean(dim=1)
        output = self.sigmoid(output)
        output = self.out(output)
        return output, None


class FIESimple(nn.Module):

    def __init__(self, vocab_size, embedding_dim, N, heads, num_class, args, max_seq_len=200):
        super().__init__()
        self.linear1 = nn.Linear(vocab_size, embedding_dim)
        self.num_frame = args.framerate * args.chunk_size
        self.similar_filter_before = MostSimpleEncoderNoPE(embedding_dim, embedding_dim, N, heads, self.num_frame//2)
        self.similar_filter_after = MostSimpleEncoderNoPE(embedding_dim, embedding_dim, N, heads, self.num_frame//2)
        self.sigmoid = nn.Sigmoid()
        self.out = nn.Linear(embedding_dim, num_class + 1)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                # nn.init.xavier_normal_(p)

    def forward(self, src, src_mask=None):
        batch_size, num_frame, num_feature = src.shape
        src = self.linear1(src)
        nb_frames_half = int(num_frame / 2)
        src_before = src[:, :nb_frames_half, :]
        src_after = src[:, nb_frames_half:, :]
        output_before, _ = self.similar_filter_before(src_before, None)
        output_after, _ = self.similar_filter_after(src_after, None)
        output = torch.cat((output_before, output_after), dim=1)

        output = output.mean(dim=1)
        output = self.sigmoid(output)
        output = self.out(output)
        return output, None


class FIE2(nn.Module):

    def __init__(self, vocab_size, embedding_dim, N, heads, num_class, max_seq_len=200):
        super().__init__()
        # if vocab_size != embedding_dim:
        self.linear1 = nn.Linear(vocab_size, embedding_dim)
        self.similar_filter_before = SimpleEncoderNoPETrue(embedding_dim, embedding_dim, N, heads,
                                                           importance_threshold=0)
        self.similar_filter_after = SimpleEncoderNoPETrue(embedding_dim, embedding_dim, N, heads,
                                                          importance_threshold=0)
        self.sigmoid = nn.Sigmoid()
        self.out = nn.Linear(embedding_dim, num_class )

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_mask=None):
        batch_size, num_frame, num_feature = src.shape
        # src = src[torch.randperm(src.size(0))]

        src = self.linear1(src)
        nb_frames_half = int(num_frame / 2)
        src_before = src[:, :nb_frames_half, :]
        src_after = src[:, nb_frames_half:, :]
        output_before, _ = self.similar_filter_before(src_before, None)
        output_after, _ = self.similar_filter_after(src_after, None)
        output = torch.cat((output_before, output_after), dim=1)

        output = output.mean(dim=1)
        output = self.sigmoid(output)
        output = self.out(output)
        return output


class FIE2_4_Highlight(nn.Module):

    def __init__(self, vocab_size, embedding_dim, N, heads, num_class, args, max_seq_len=200):
        super().__init__()
        # if vocab_size != embedding_dim:
        self.linear1 = nn.Linear(vocab_size, embedding_dim)
        self.similar_filter_before = SimpleEncoderNoPETrue(embedding_dim, embedding_dim, N, heads,
                                                           importance_threshold=args.i_t)
        self.similar_filter_after = SimpleEncoderNoPETrue(embedding_dim, embedding_dim, N, heads,
                                                          importance_threshold=args.i_t)
        self.sigmoid = nn.Sigmoid()
        self.out = nn.Linear(embedding_dim, num_class + 1)
        self.N = N
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_mask=None):
        batch_size, num_frame, num_feature = src.shape
        src = self.linear1(src)
        nb_frames_half = int(num_frame / 2)
        src_before = src[:, :nb_frames_half, :]
        src_after = src[:, nb_frames_half:, :]
        output_before, enc_attn_before = self.similar_filter_before(src_before, None)
        output_after, enc_attn_after = self.similar_filter_after(src_after, None)

        attn_score_before = visual_attn(enc_attn_before, n_layer=self.N)
        attn_score_after = visual_attn(enc_attn_after, n_layer=1)
        attn = torch.cat((attn_score_before, attn_score_after), dim=1)

        output = torch.cat((output_before, output_after), dim=1)

        output = output.mean(dim=1)
        output = self.sigmoid(output)
        output = self.out(output)
        return output, attn


class FIE2PE(nn.Module):

    def __init__(self, vocab_size, embedding_dim, N, heads, num_class, args, max_seq_len=200):
        super().__init__()
        # if vocab_size != embedding_dim:
        self.pe = NewPositionalEncoder(vocab_size, max_seq_len)
        self.linear1 = nn.Linear(vocab_size, embedding_dim)
        self.similar_filter_before = SimpleEncoderNoPETrue(embedding_dim, embedding_dim, N, heads)
        self.similar_filter_after = SimpleEncoderNoPETrue(embedding_dim, embedding_dim, N, heads)
        self.sigmoid = nn.Sigmoid()
        self.out = nn.Linear(embedding_dim, num_class + 1)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_mask=None):
        batch_size, num_frame, num_feature = src.shape
        src = self.pe(src)
        src = self.linear1(src)
        nb_frames_half = int(num_frame / 2)
        src_before = src[:, :nb_frames_half, :]
        src_after = src[:, nb_frames_half:, :]
        output_before, _ = self.similar_filter_before(src_before, None)
        output_after, _ = self.similar_filter_after(src_after, None)
        output = torch.cat((output_before, output_after), dim=1)

        output = output.mean(dim=1)
        output = self.sigmoid(output)
        output = self.out(output)
        return output, None


class Args:
    def __init__(self):
        self.similar_threshold = 0
        self.chunk_size = 20
        self.framerate = 2
        self.use_spot_loss = True
        self.pooling_module = "NetVLAD"


def visual_attn(attn, n_layer=1):
    # layer(6), 1, head(8), frame, frame
    # attn = attn[0]
    # n_batch = attn.shape[0]
    # ATTN = []
    # for z in range(n_batch):
    #     Attn = attn[z]
    #     ATTN.append(Attn)

    # layer(6), head(8), frame, frame
    tensor_attn = torch.stack(attn, dim=0)
    # tuple(layer(6)), 1, head(8), frame, frame
    # ATTN = tuple(ATTN)
    # 全headのattention weightを加算
    tensor_attn = torch.sum(tensor_attn, dim=3)
    # ノードで加算
    # tensor_attn = torch.sum(tensor_attn, dim=1)
    # layerで平均
    tensor_attn = torch.prod(tensor_attn, dim=0)
    # print(tensor_attn)  # 20 seconds

    # 秒数確認
    # torch.argsort(-tensor_attn)

    # TOKEN = [str(i) for i in range(0, chunk_size)]

    # TOKEN = [str(i) for i in range(0, 120)]
    # head_view(ATTN, TOKEN)
    return tensor_attn


if __name__ == '__main__':
    n_batch = 64
    n_frame = 30
    n_feature = 2048
    args = Args()
    # model = BaseFilter()
    # model = MyBaseTransformer(512, 512, 6, 8, 17, args)
    # model = MyTransformer(512, 512, 6, 8, 17, args)
    # model = BestTransformer20_4_atten(512, 512, 1, 4, 17, args)
    model = BestTransformer20(512, 512, 1, 4, 17, args)
    input = torch.randn(n_batch, n_frame, n_feature)

    output, output_1 = model(input)
    print(output.shape)
    print(output_1.shape)
    # print(torch.ones_like(output_1).shape)
    print([i.shape for i in output])
