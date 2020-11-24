import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchvision.models import vgg16_bn

from base.base_net_structure import BaseNetStructure
from utils.net_modules_utils import Conv


class InitWeightMixin:
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


class DeconvBnReLU(nn.Module, InitWeightMixin):
    def __init__(self, in_n, out_n, is_init=True):
        super(DeconvBnReLU, self).__init__()
        self.model = nn.Sequential(nn.ConvTranspose2d(
            in_n, out_n, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_n),
            nn.ReLU(inplace=True),
        )
        if is_init:
            self.init_weight()

    def forward(self, x):
        return self.model(x)


class ConvBnReLU(Conv, InitWeightMixin):
    def __init__(self, in_n, out_n, is_init=False, kernel_size=3, stride=1, padding=1, bias=False):
        super(ConvBnReLU, self).__init__(in_n, out_n, kernel_size, stride, padding, bias)
        self.add_module("BN", nn.BatchNorm2d(out_n, eps=1e-5, momentum=0.999))
        self.add_module("ReLU", nn.ReLU(inplace=True))
        if is_init:
            self.init_weight()


class FCNVgg16(nn.Module):
    def __init__(self, n_out=4, is_init=False):
        """
        网络初始化，特点输出结果的图像大小和输入图像的是一样的
        :param n_out: 输出结果的频道数。
        """
        super(FCNVgg16, self).__init__()
        vgg = vgg16_bn(pretrained=False)
        self.encoder_1 = vgg.features[:7]
        self.encoder_2 = vgg.features[7:14]
        self.encoder_3 = vgg.features[14:24]
        self.encoder_4 = vgg.features[24:34]
        self.encoder_5 = vgg.features[34:]
        self.decoder_1 = DeconvBnReLU(512, 512, is_init)
        self.decoder_2 = DeconvBnReLU(512, 256, is_init)
        self.decoder_3 = DeconvBnReLU(256, 128, is_init)
        self.decoder_4 = DeconvBnReLU(128, 64, is_init)
        self.decoder_5 = DeconvBnReLU(64, n_out, is_init)

    def forward(self, x):
        out_1 = self.encoder_1(x)
        out_2 = self.encoder_2(out_1)
        out_3 = self.encoder_3(out_2)
        out_4 = self.encoder_4(out_3)
        out_5 = self.encoder_5(out_4)

        decoder_1 = self.decoder_1(out_5)
        decoder_2 = self.decoder_2(decoder_1 + out_4)
        decoder_3 = self.decoder_3(decoder_2 + out_3)
        decoder_4 = self.decoder_4(decoder_3 + out_2)
        out = self.decoder_5(decoder_4 + out_1)
        return out


class FCN4Edge_3(nn.Module):
    def __init__(self, n_out=4, is_init=False):
        """
        网络初始化，特点输出结果的图像大小和输入图像的是一样的
        :param n_out: 输出结果的频道数。
        """
        super(FCN4Edge_3, self).__init__()
        self.encoder_0 = ConvBnReLU(2, 3, is_init=is_init)

        vgg = vgg16_bn(pretrained=False)

        self.encoder0_1 = vgg.features[:7]
        self.encoder0_2 = vgg.features[7:14]
        self.encoder0_3 = vgg.features[14:24]
        self.encoder0_4 = vgg.features[24:34]
        self.encoder0_5 = vgg.features[34:]

        self.encoder1_1 = vgg.features[:7]
        self.encoder1_2 = vgg.features[7:14]
        self.encoder1_3 = vgg.features[14:24]
        self.encoder1_4 = vgg.features[24:34]
        self.encoder1_5 = vgg.features[34:]

        self.decoder_1 = DeconvBnReLU(512, 512, is_init)
        self.decoder_2 = DeconvBnReLU(512, 256, is_init)
        self.decoder_3 = DeconvBnReLU(256, 128, is_init)
        self.decoder_4 = DeconvBnReLU(128, 64, is_init)
        self.decoder_5 = DeconvBnReLU(64, n_out, is_init)

    def forward(self, x):
        image = x[:, :3]
        other = x[:, 3:]

        out_0 = self.encoder_0(other)

        out_1 = self.encoder0_1(out_0)
        out_2 = self.encoder0_2(out_1)
        out_3 = self.encoder0_3(out_2)
        out_4 = self.encoder0_4(out_3)
        out_5 = self.encoder0_5(out_4)

        image_out_1 = self.encoder1_1(image)
        image_out_2 = self.encoder1_2(image_out_1)
        image_out_3 = self.encoder1_3(image_out_2)
        image_out_4 = self.encoder1_4(image_out_3)
        image_out_5 = self.encoder1_5(image_out_4)

        decoder_1 = self.decoder_1(out_5 + image_out_5)
        decoder_2 = self.decoder_2(decoder_1 + out_4 + image_out_4)
        decoder_3 = self.decoder_3(decoder_2 + out_3 + image_out_3)
        decoder_4 = self.decoder_4(decoder_3 + out_2 + image_out_2)
        out = self.decoder_5(decoder_4 + out_1 + image_out_1)
        return out


class FCN4Edge_2(nn.Module):
    def __init__(self, n_out=4, is_init=False):
        """
        网络初始化，特点输出结果的图像大小和输入图像的是一样的
        :param n_out: 输出结果的频道数。
        """
        super(FCN4Edge_2, self).__init__()
        self.encoder_0 = ConvBnReLU(2, 3, is_init=is_init)

        vgg = vgg16_bn(pretrained=False)

        self.encoder0_1 = vgg.features[:7]
        self.encoder0_2 = vgg.features[7:14]
        self.encoder0_3 = vgg.features[14:24]
        self.encoder0_4 = vgg.features[24:34]
        self.encoder0_5 = vgg.features[34:]

        self.encoder1_1 = vgg.features[:7]
        self.encoder1_2 = vgg.features[7:14]
        self.encoder1_3 = vgg.features[14:24]
        self.encoder1_4 = vgg.features[24:34]
        self.encoder1_5 = vgg.features[34:]

        self.decoder_1 = DeconvBnReLU(1024, 512, is_init)
        self.decoder_2 = DeconvBnReLU(1536, 256, is_init)
        self.decoder_3 = DeconvBnReLU(768, 128, is_init)
        self.decoder_4 = DeconvBnReLU(384, 64, is_init)
        self.decoder_5 = DeconvBnReLU(192, n_out, is_init)

    def forward(self, x):
        image = x[:, :3]
        other = x[:, 3:]

        # print(id(self.encoder0_1) == id(self.encoder1_1))

        # 编码器部分
        out_0 = self.encoder_0(other)

        out_1 = self.encoder0_1(out_0)
        out_2 = self.encoder0_2(out_1)
        out_3 = self.encoder0_3(out_2)
        out_4 = self.encoder0_4(out_3)
        out_5 = self.encoder0_5(out_4)

        image_out_1 = self.encoder1_1(image)
        image_out_2 = self.encoder1_2(image_out_1)
        image_out_3 = self.encoder1_3(image_out_2)
        image_out_4 = self.encoder1_4(image_out_3)
        image_out_5 = self.encoder1_5(image_out_4)

        # 解码器部分
        decoder_1 = self.decoder_1(torch.cat([out_5, image_out_5], dim=1))
        decoder_2 = self.decoder_2(torch.cat([decoder_1, out_4, image_out_4], dim=1))
        decoder_3 = self.decoder_3(torch.cat([decoder_2, out_3, image_out_3], dim=1))
        decoder_4 = self.decoder_4(torch.cat([decoder_3, out_2, image_out_2], dim=1))
        out = self.decoder_5(torch.cat([decoder_4, out_1, image_out_1], dim=1))
        return out


class FCN4Edge(nn.Module, BaseNetStructure):
    def __init__(self, n_in=5, n_out=4, is_init=False):
        """
        网络初始化，特点输出结果的图像大小和输入图像的是一样的
        :param n_out: 输出结果的频道数。
        """
        super(FCN4Edge, self).__init__()
        self.encoder_0 = ConvBnReLU(n_in, 3, is_init=is_init)

        vgg = vgg16_bn(pretrained=False)

        self.encoder_1 = vgg.features[:7]
        self.encoder_2 = vgg.features[7:14]
        self.encoder_3 = vgg.features[14:24]
        self.encoder_4 = vgg.features[24:34]
        self.encoder_5 = vgg.features[34:]

        self.decoder_1 = DeconvBnReLU(512, 512, is_init)
        self.decoder_2 = DeconvBnReLU(512, 256, is_init)
        self.decoder_3 = DeconvBnReLU(256, 128, is_init)
        self.decoder_4 = DeconvBnReLU(128, 64, is_init)
        self.decoder_5 = DeconvBnReLU(64, n_out, is_init)

    def forward(self, x):
        out_0 = self.encoder_0(x)
        out_1 = self.encoder_1(out_0)
        out_2 = self.encoder_2(out_1)
        out_3 = self.encoder_3(out_2)
        out_4 = self.encoder_4(out_3)
        out_5 = self.encoder_5(out_4)

        decoder_1 = self.decoder_1(out_5)
        decoder_2 = self.decoder_2(decoder_1 + out_4)
        decoder_3 = self.decoder_3(decoder_2 + out_3)
        decoder_4 = self.decoder_4(decoder_3 + out_2)
        out = self.decoder_5(decoder_4 + out_1)
        return out


class FCNRes(nn.Module):

    def __init__(self, n_out=1, is_init=False):
        super(FCNRes, self).__init__()

        self.encoder = models.resnet34(pretrained=True)
        self.conv_bn_relu0 = ConvBnReLU(512, 512, kernel_size=3, stride=2)

        self.decoder_1 = DeconvBnReLU(512, 512, is_init)
        self.decoder_2 = DeconvBnReLU(512, 256, is_init)
        self.decoder_3 = DeconvBnReLU(256, 128, is_init)
        self.decoder_4 = DeconvBnReLU(128, 64, is_init)
        self.decoder_5 = DeconvBnReLU(64, n_out, is_init)

    def forward(self, x):
        out = self.encoder.conv1(x)
        out = self.encoder.bn1(out)
        out = F.relu(out)
        out_1 = self.encoder.layer1(out)
        out_2 = self.encoder.layer2(out_1)
        out_3 = self.encoder.layer3(out_2)
        out_4 = self.encoder.layer4(out_3)
        out_5 = self.conv_bn_relu0(out_4)

        decoder_1 = self.decoder_1(out_5)
        decoder_2 = self.decoder_2(decoder_1 + out_4)
        decoder_3 = self.decoder_3(decoder_2 + out_3)
        decoder_4 = self.decoder_4(decoder_3 + out_2)
        out = self.decoder_5(decoder_4 + out_1)
        return out


class FCNRes50(nn.Module):

    def __init__(self, num_classes=1, is_init=False):
        super(FCNRes50, self).__init__()
        import torchvision.models as models

        self.encoder = models.resnet50(pretrained=False)

        self.decoder_1 = DeconvBnReLU(2048, 1024, is_init)
        self.decoder_2 = DeconvBnReLU(1024, 512, is_init)
        self.decoder_3 = DeconvBnReLU(512, 256, is_init)
        self.decoder_4 = DeconvBnReLU(256, 64, is_init)
        self.decoder_5 = DeconvBnReLU(64, num_classes, is_init)

    def forward(self, x):
        out = self.encoder.conv1(x)
        print(out.shape)
        out = self.encoder.bn1(out)
        print(out.shape)
        out_0 = self.encoder.relu(out)
        print(out_0.shape)
        out_0 = self.encoder.maxpool(out_0)
        print('out_0', out_0.shape)
        out_1 = self.encoder.layer1(out_0)
        print('out_1', out_1.shape)
        out_2 = self.encoder.layer2(out_1)
        print('out_2', out_2.shape)
        out_3 = self.encoder.layer3(out_2)
        print('out_3', out_3.shape)
        out_4 = self.encoder.layer4(out_3)
        print('out_4', out_4.shape)
        decoder_1 = self.decoder_1(out_4)
        print('decoder_1', decoder_1.shape)
        decoder_2 = self.decoder_2(decoder_1 + out_3)
        print('decoder_2', decoder_2.shape)
        decoder_3 = self.decoder_3(decoder_2 + out_2)
        print('decoder_3', decoder_3.shape)
        decoder_4 = self.decoder_4(decoder_3 + out_1)
        print('decoder_4', decoder_4.shape)
        out = self.decoder_5(decoder_4 + out)
        print("out", out.shape)
        return out


class FeatureExtractor(nn.Module):
    """提取特定层的输出，依靠层的名字来匹配"""

    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    # 自己修改forward函数
    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            if name is "fc": x = x.view(x.size(0), -1)
            x = module(x)
            if name in self.extracted_layers:
                outputs.append(x)
        return outputs
