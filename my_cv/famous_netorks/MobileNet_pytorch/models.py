import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Sequential):
    def __init__(self, in_n, out_n, kernel_size=3, stride=1, padding=1, **kwargs):
        super(Conv, self).__init__()
        self.add_module("Conv", nn.Conv2d(in_n, out_n, kernel_size=kernel_size,
                                          stride=stride, padding=padding, bias=False,
                                          **kwargs))


class ConvBN(Conv):
    def __init__(self, in_n, out_n, kernel_size=3, stride=1, padding=1, **kwargs):
        super(ConvBN, self).__init__(in_n, out_n, kernel_size, stride, padding, **kwargs)
        self.add_module("BN", nn.BatchNorm2d(out_n))


class ConvBNReLU(ConvBN):
    def __int__(self, in_n, out_n, kernel_size=3, stride=1, padding=1, **kwargs):
        super(ConvBNReLU, self).__init__(in_n, out_n, kernel_size, stride, padding, **kwargs)
        self.add_module("ReLU", nn.ReLU(inplace=True))


class DepthPointConv(nn.Module):
    """深度卷积和逐点卷积"""

    def __init__(self, in_chans, out_chans, stride=1, is_relu=True):
        super(DepthPointConv, self).__init__()
        self.conv_bn_relu = ConvBNReLU(in_chans, in_chans, stride=stride, groups=in_chans)
        self.conv_bn = ConvBN(in_chans, out_chans, kernel_size=1, padding=0)
        self.is_relu = is_relu

    def forward(self, x):
        out = self.conv_bn_relu(x)
        if self.is_relu:
            # 如果需要relu的话，就经过最后添加relu层
            return F.relu(self.conv_bn(out), inplace=True)
        else:
            return self.conv_bn(out)


class MobileNetV1(nn.Module):
    # [(conv num_channels, stirde),(conv planes, stirde)，(conv planes, stirde)....]
    cfg = [(64, 1), (128, 2), (128, 1), (256, 2), (256, 1), (512, 2),
           (512, 1), (512, 1), (512, 1), (512, 1), (512, 1), (1024, 2),
           (1024, 1)]

    def __init__(self, num_classes=10):
        super(MobileNetV1, self).__init__()
        # 网络的第一层是普通的卷积
        self.conv_bn_relu = ConvBNReLU(3, 32)
        self.layers = self._make_layers(in_chans=32)
        self.classifier = nn.Linear(1024, num_classes)

    def _make_layers(self, in_chans):
        layers = []
        for x in self.cfg:
            out_chans = x[0]
            stride = x[1]
            layers.append(DepthPointConv(in_chans, out_chans, stride))
            # 上个输出通道数变成下一个输入通道数
            in_chans = out_chans
        # 最后变成一个Sequential
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv_bn_relu(x)
        out = self.layers(out)
        # 平均池化
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


class PointDepthPointConv(nn.Module):
    """先点卷积，然后深度卷积，最后再点卷积"""

    def __init__(self, in_chans, out_chans, expansion, stride):
        super(PointDepthPointConv, self).__init__()
        self.stride = stride
        mid_chans = expansion * in_chans
        self.conv_bn_relu = ConvBNReLU(in_chans, mid_chans, kernel_size=1, padding=0)
        self.depth_point_wise = DepthPointConv(mid_chans, out_chans, stride=stride, is_relu=False)

        if stride == 1 and in_chans != out_chans:
            # 如果stride等于1
            self.shortcut = ConvBN(in_chans, out_chans, kernel_size=1, padding=0)
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.conv_bn_relu(x)
        out = self.depth_point_wise(out)
        if self.stride == 1:
            return out + self.shortcut(x)
        else:
            return out


class MobileNetV2(nn.Module):
    # （通道扩大的倍数，输出通道数，重复次数，stride）
    cfg = [(1, 16, 1, 1),
           (6, 24, 2, 1),  # 为了适应CIFAR10数据集的大小，把论文中的stride=2改为stride=1了
           (6, 32, 3, 2),
           (6, 64, 4, 2),
           (6, 96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=10):
        super(MobileNetV2, self).__init__()
        # 论文中的stride是2，但是为了使用CIFAR10数据库，改为1
        self.conv_bn_relu1 = ConvBNReLU(3, 32)
        self.layers = self._make_layers(in_chans=32)
        self.conv_bn_relu2 = ConvBNReLU(320, 1280, kernel_size=1, padding=0)
        self.classifier = nn.Linear(1280, num_classes)

    def _make_layers(self, in_chans):
        layers = []
        for expansion, out_chans, num_blocks, stride in self.cfg:
            # 除了第一个stride是设定好的，其他的stride都是1
            strides = [stride] + [1] * (num_blocks - 1)
            for stride in strides:
                layers.append(PointDepthPointConv(in_chans, out_chans, expansion, stride))
                # 上一个通道数变为下一个通道数
                in_chans = out_chans
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv_bn_relu1(x)
        out = self.layers(out)
        out = self.conv_bn_relu2(out)
        # 如果是使用CIFAR10数据集,kernel_size要设置为4
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


if __name__ == '__main__':
    # 经测试可以跑通
    model = MobileNetV1()
    x = torch.randn(8, 3, 32, 32)
    y = model(x)
    print(y.size())

    # 懒得改代码了，就使用relu代替MobileNet的ReLU6了
    model = MobileNetV2()
    x = torch.randn(8, 3, 32, 32)
    y = model(x)
    print(y.size())
