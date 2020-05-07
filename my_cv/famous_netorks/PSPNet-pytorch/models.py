import torch
import torch.nn.functional as F
from Resnet import resnet_50 as resnet_50
from torch import nn


class ConvBNReLU(nn.Module):
    def __init__(self, in_n, out_n, kernel_size=3, stride=1, padding=1):
        super(ConvBNReLU, self).__init__()
        self.conv_1 = nn.Conv2d(in_n, out_n, kernel_size, stride, padding, bias=False)
        self.bn_1 = nn.BatchNorm2d(out_n)

    def forward(self, x):
        out = self.conv_1(x)
        out = self.bn_1(out)
        return F.relu(out, inplace=True)


class SppBlock(nn.Module):
    def __init__(self, size, in_n, out_n=512):
        super(SppBlock, self).__init__()
        # 设置池化层输出的大小
        self.size = size
        self.conv_block = ConvBNReLU(in_n, out_n, kernel_size=1, padding=0)

    def forward(self, x):
        size = x.shape[2]
        # 调整输出大小的池化
        out = F.adaptive_avg_pool2d(x, output_size=(self.size, self.size))
        out = self.conv_block(out)
        # 上采样
        out = F.interpolate(out, size=size, mode='bilinear', align_corners=True)
        return out


class SPP(nn.Module):
    def __init__(self, in_n):
        super(SPP, self).__init__()
        self.spp_1 = SppBlock(size=1, in_n=in_n)
        self.spp_2 = SppBlock(size=2, in_n=in_n)
        self.spp_3 = SppBlock(size=3, in_n=in_n)
        self.spp_4 = SppBlock(size=6, in_n=in_n)

    def forward(self, x):
        x_1 = self.spp_1(x)
        x_2 = self.spp_2(x)
        x_3 = self.spp_3(x)
        x_4 = self.spp_4(x)
        # 把每个spp的输出和输入x拼接起来
        out = torch.cat([x, x_1, x_2, x_3, x_4], dim=1)
        return out


class PSPNet(nn.Module):
    def __init__(self, res_net_layers=50, dropout=0.1, n_classes=21,
                 criterion=nn.CrossEntropyLoss(ignore_index=255)):
        """
        建立PSP net
        :param res_net_layers: ResNet的层数
        :param dropout: dropout的层数
        :param n_classes: 最后分类的个数
        :param criterion:  使用损失函数
        :return:
        """
        super(PSPNet, self).__init__()
        self.criterion = criterion

        if res_net_layers == 50:
            resnet = resnet_50()
        else:
            # todo 增加其他的resnet
            resnet = resnet_50()

        self.layer_0 = nn.Sequential(resnet.conv_bn_relu_1,
                                     resnet.conv_bn_relu_2,
                                     resnet.conv_bn_relu_3,
                                     resnet.maxpool)

        self.layer_1, self.layer_2, self.layer_3, self.layer_4 = \
            resnet.layer_1, resnet.layer_2, resnet.layer_3, resnet.layer_4
        for name, model in self.layer_3.named_modules():
            # print('n is ',name)
            # print('m is ',model)
            # print("========================")
            if 'conv_bn_relu_2.Conv' in name:
                model.dilation, model.padding, model.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in name:
                model.stride = (1, 1)
        for n, m in self.layer_4.named_modules():
            # print('n is ',n)
            # print('m is ',m)
            # print("========================")
            if 'conv_bn_relu_2.Conv' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        self.spp = SPP(in_n=2048)
        # 分类器
        self.cls = nn.Sequential(ConvBNReLU(2048 + 512 * 4, 512),
                                 nn.Dropout2d(p=dropout),
                                 nn.Conv2d(512, n_classes, kernel_size=1))
        # 用于输出中间结果，计算辅助损失函数
        if self.training:
            self.aux = nn.Sequential(
                ConvBNReLU(1024, 256),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(256, n_classes, kernel_size=1)
            )

    def forward(self, x, y=None):
        # 图像的尺寸
        size = x.shape[2:]
        out = self.layer_0(x)
        out = self.layer_1(out)
        out = self.layer_2(out)
        temp = self.layer_3(out)
        out = self.layer_4(temp)
        out = self.spp(out)
        out = self.cls(out)
        # 根据输入图像的大小输出结果
        # todo 可以增加一个调节输出大小的参数，用于放大缩小
        out = F.interpolate(out, size, mode='bilinear', align_corners=True)

        if self.training:
            # 训练的时候有两个输出
            # 要计算两个损失函数
            aux = self.aux(temp)
            if self.zoom_factor != 1:
                aux = F.interpolate(aux, size, mode='bilinear', align_corners=True)
            main_loss = self.criterion(out, y)
            aux_loss = self.criterion(aux, y)
            return out.max(1)[1], main_loss, aux_loss
        else:
            return out


if __name__ == '__main__':
    # 经测试结构没有问题
    input = torch.rand(4, 3, 473, 473)
    model = PSPNet()
    model.eval()
    output = model(input)
    print('PSPNet', output.size())
