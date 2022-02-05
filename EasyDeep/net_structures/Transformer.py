#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/2/3 18:37
# @Author  : strawsyz
# @File    : Transformer.py
# @desc:


import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def attention(q, k, v, d_k, mask=None, dec_mask=False):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        if dec_mask:
            mask = mask.view(mask.size(0), 1, mask.size(1), mask.size(2))
        else:
            mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)

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


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, embedding_dim):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.d_k = embedding_dim // heads
        self.h = heads

        self.q_linear = nn.Linear(embedding_dim, embedding_dim)
        self.v_linear = nn.Linear(embedding_dim, embedding_dim)
        self.k_linear = nn.Linear(embedding_dim, embedding_dim)

        self.out = nn.Linear(embedding_dim, embedding_dim)

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


class PositionalEncoder(nn.Module):
    def __init__(self, embedding_dim, max_seq_len=200):
        super().__init__()
        self.embedding_dim = embedding_dim
        pe = torch.zeros(max_seq_len, embedding_dim)
        for pos in range(max_seq_len):
            for i in range(0, embedding_dim, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / embedding_dim)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / embedding_dim)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * math.sqrt(self.embedding_dim)
        seq_len = x.size(1)
        pe = Variable(self.pe[:, :seq_len], requires_grad=False)
        if x.is_cuda:
            pe.cuda()
        x = x + pe
        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, N, heads):
        super().__init__()
        self.N = N
        self.pe = PositionalEncoder(embedding_dim)
        self.ln = nn.Linear(vocab_size, embedding_dim)
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


class Transformer1E(nn.Module):
    """best model on 05/01"""

    def __init__(self, vocab_size, embedding_dim, N, heads, num_class):
        super().__init__()
        # self.feature_extractor = nn.Linear(2048, 512)
        self.encoder = Encoder(vocab_size, embedding_dim, N, heads)
        self.sigmoid = nn.Sigmoid()
        self.out = nn.Linear(embedding_dim, num_class + 1)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, src, similarity=None, src_mask=None):
        output, enc_attn = self.encoder(src, src_mask)
        output = output.mean(dim=1)
        output = self.sigmoid(output)
        output = self.dropout(output)
        output = self.out(output)
        return output


class Transformer2E(nn.Module):

    def __init__(self, vocab_size, embedding_dim, N, heads, num_class):
        super().__init__()
        self.feature_extractor = nn.Linear(vocab_size, 512)
        self.encoder_before = Encoder(512, embedding_dim, N, heads)
        self.encoder_after = Encoder(512, embedding_dim, N, heads)
        self.sigmoid = nn.Sigmoid()
        self.out = nn.Linear(embedding_dim, num_class)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, src, src_mask=None):
        batch_size, num_frame, num_feature = src.shape
        src = self.feature_extractor(src)
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
        return output


if __name__ == '__main__':
    model = Transformer2E(2048, 512, 6, 8, 101)

    model.eval()
    n_batch = 32
    n_frame = 10
    n_feature = 2048
    # src_mask = create_masks(10, 512)
    # data = torch.from_numpy(data)
    # data = torch.randn(n_batch, n_frame, n_feature)
    # src_mask = torch.randn(n_batch, 1, 10)
    data = torch.randn(n_batch, n_frame, n_feature)
    print(data.shape)
    # print(src_mask.shape)
    out = model(data)
    print(out.shape)
