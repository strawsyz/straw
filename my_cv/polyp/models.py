# -*- coding: utf-8 -*-

import torch
from torch import nn
from torchvision.models import vgg16_bn


class Deconv(nn.Module):
    # 上采样用的编码器
    def __init__(self, in_n, out_n):
        super(Deconv, self).__init__()
        self.model = nn.Sequential(nn.ConvTranspose2d(
            in_n, out_n, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_n),
            nn.ReLU(inplace=True),
        )
        # 初始化网络的参数
        self.init_weight()

    def forward(self, x):
        return self.model(x)

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class FCN(nn.Module):
    def __init__(self, n_out=4):
        """
        网络初始化，特点输出结果的图像大小和输入图像的是一样的
        :param n_out: 输出结果的频道数。
        """
        super(FCN, self).__init__()
        # 在VGG的基础上建立，使用VGG的结构
        vgg = vgg16_bn(pretrained=True)

        # 编码器
        self.encoder_1 = vgg.features[:7]
        self.encoder_2 = vgg.features[7:14]
        self.encoder_3 = vgg.features[14:24]
        self.encoder_4 = vgg.features[24:34]
        self.encoder_5 = vgg.features[34:]
        # 解码器
        self.decoder_1 = Deconv(512, 512)
        self.decoder_2 = Deconv(512, 256)
        self.decoder_3 = Deconv(256, 128)
        self.decoder_4 = Deconv(128, 64)
        self.decoder_5 = Deconv(64, n_out)

    def forward(self, x):
        # 编码器部分
        out_1 = self.encoder_1(x)
        out_2 = self.encoder_2(out_1)
        out_3 = self.encoder_3(out_2)
        out_4 = self.encoder_4(out_3)
        out_5 = self.encoder_5(out_4)

        # 解码器部分
        decoder_1 = self.decoder_1(out_5)
        decoder_2 = self.decoder_2(decoder_1 + out_4)
        decoder_3 = self.decoder_3(decoder_2 + out_3)
        # 输出中间结果，对中间结果也进行优化
        decoder_4 = self.decoder_4(decoder_3 + out_2)
        out = self.decoder_5(decoder_4 + out_1)
        return out, decoder_4


def network():
    # 输出一个通道
    return FCN(n_out=1)


if __name__ == '__main__':
    # 测试数据能正常跑通
    x = torch.randn(1, 3, 224, 224)
    net = FCN()
    out = net(x)
    print(out.shape)
