# -*- coding: utf-8 -*-
import torch
from torch import nn


class BasicBlock(nn.Module):
    def __init__(self, in_n, out_n, kernel_size,
                 stride=1, padding=0, groups=1, bias=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_n, out_n, kernel_size, stride, padding, groups=groups, bias=bias)
        self.bn1 = nn.BatchNorm2d(out_n)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_n, out_n, kernel_size, 1, padding, groups=groups, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_n)
        self.downsample = None
        if stride > 1:
            # 如果第二个卷积层上采样了，这边就需要下采样一次
            self.downsample = nn.Sequential(
                nn.Conv2d(in_n, out_n, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_n))

    def forward(self, x):
        if self.downsamplele:
            # 如果有降采样层就将降采样x,作为残差部分
            residual = self.downsample(x)
        else:
            # 如果没有就直接使用x作为残差部分
            residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # 添加残差部分
        out += residual

        out = self.relu(out)
        return out


class Encoder(nn.Module):
    ## todo 也可以使用ResNet18作为编码器的部分
    def __init__(self, in_n, out_n, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(Encoder, self).__init__()

        self.block1 = BasicBlock(in_n, out_n, kernel_size, stride, padding, groups, bias)
        self.block2 = BasicBlock(out_n, out_n, kernel_size, 1, padding, groups, bias)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)

        return out


class Decoder(nn.Module):
    def __init__(self, in_n, out_n, kernel_size, stride=1, padding=0,
                 output_padding=0, bias=False):
        super(Decoder, self).__init__()
        self.conv_block_1 = nn.Sequential(nn.Conv2d(in_n, in_n // 4, 1, 1, 0, bias=bias),
                                          nn.BatchNorm2d(in_n // 4),
                                          nn.ReLU(inplace=True),
                                          )
        self.conv_tp_block = nn.Sequential(nn.ConvTranspose2d(in_n // 4, in_n // 4, kernel_size,
                                                              stride, padding, output_padding, bias=bias),
                                           nn.BatchNorm2d(in_n // 4),
                                           nn.ReLU(inplace=True),
                                           )
        self.conv_block_2 = nn.Sequential(nn.Conv2d(in_n // 4, out_n, 1, 1, 0, bias=bias),
                                          nn.BatchNorm2d(out_n),
                                          nn.ReLU(),
                                          )

    def forward(self, x):
        out = self.conv_block_1(x)
        out = self.conv_tp_block(out)
        return self.conv_block_2(out)


class LinkNet(nn.Module):
    def __ini__(self, n_classes=21):
        super(LinkNet, self).__init__()
        self.conv_block_1 = nn.Sequential(nn.Conv2d(3, 64, 7, 2, 3, bias=False),
                                          nn.BatchNorm2d(64),
                                          nn.ReLU(inplace=True),
                                          nn.MaxPool2d(3, 2, 1),
                                          )

        # 编码器
        self.encoder_1 = Encoder(64, 64, 3, 1, 1)
        self.encoder_2 = Encoder(64, 128, 3, 2, 1)
        self.encoder_3 = Encoder(128, 256, 3, 2, 1)
        self.encoder_4 = Encoder(256, 512, 3, 2, 1)
        # 解码器
        self.decoder_1 = Decoder(64, 64, 3, 1, 1, 0)
        self.decoder_2 = Decoder(128, 64, 3, 2, 1, 1)
        self.decoder_3 = Decoder(256, 128, 3, 2, 1, 1)
        self.decoder_4 = Decoder(512, 256, 3, 2, 1, 1)
        # 分类器
        self.conv_tp_block_1 = nn.Squential(nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
                                            nn.BatchNorm2d(32),
                                            nn.ReLU(inplace=True),
                                            )
        self.conv_block_2 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1),
                                          nn.BatchNorm2d(32),
                                          nn.ReLU(inplace=True),
                                          )
        self.conv_tp_block_2 = nn.Sequential(nn.ConvTranspose2d(32, n_classes, 2, 2, 0),
                                             nn.LogSoftmax(dim=1),
                                             )

    def forward(self, x):
        out = self.conv_block_1(x)
        # 编码器
        e1 = self.encoder_1(out)
        e2 = self.encoder_2(e1)
        e3 = self.encoder_3(e2)
        e4 = self.encoder_4(e3)
        # 解码器
        d4 = e3 + self.decoder_4(e4)
        d3 = e2 + self.decoder_3(d4)
        d2 = e1 + self.decoder_2(d3)
        d1 = x + self.decoder_1(d2)
        # 分类器
        out = self.conv_tp_block_1(d1)
        out = self.conv_block_2(out)
        out = self.conv_tp_block_2(out)
        return out

# if __name__ == '__main__':
#     LinkNet()