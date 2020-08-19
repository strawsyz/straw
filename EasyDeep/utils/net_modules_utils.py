from torch import nn


class BNReLUConv(nn.Sequential):
    def __init__(self, in_n, out_n, **kwargs):
        super(BNReLUConv, self).__init__()
        self.add_module('BN', nn.BatchNorm2d(in_n))
        self.add_module("ReLU", nn.ReLU(inplace=True))
        self.add_module("Conv", nn.Conv2d(in_n, out_n, bias=False, **kwargs))


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Conv(nn.Sequential):
    def __init__(self, in_n, out_n, kernel_size=3, stride=1, padding=1, bias=False, **kwargs):
        super(Conv, self).__init__()
        self.add_module("Conv2d",
                        nn.Conv2d(in_n, out_n, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias,
                                  **kwargs))


class ConvReLU(Conv):
    def __init__(self, in_n, out_n, kernel_size=3, stride=1, padding=1, bias=False):
        super(ConvReLU, self).__init__(in_n, out_n, kernel_size, stride, padding, bias)
        self.add_module("ReLU", nn.ReLU(inplace=True))


class LinearReLU(nn.Sequential):
    def __init__(self, in_n, out_n):
        super(LinearReLU, self).__init__()
        self.add_module("Linear", nn.Linear(in_n, out_n))
        self.add_module("ReLU", nn.ReLU(inplace=True))


class ConvBn(Conv):
    def __init__(self, in_n, out_n, kernel_size=3, stride=1, padding=1, bias=False, **kwargs):
        super(ConvBn, self).__init__(in_n, out_n, kernel_size, stride, padding, bias, **kwargs)
        self.add_module("BN", nn.BatchNorm2d(out_n, eps=1e-5, momentum=0.999))


class ConvBnReLU(ConvBn):
    def __init__(self, in_n, out_n, kernel_size=3, stride=1, padding=1, bias=False, **kwargs):
        super(ConvBnReLU, self).__init__(in_n, out_n, kernel_size, stride, padding, bias, **kwargs)
        self.add_module("ReLU", nn.ReLU(inplace=True))


class DeconvBNReLU(nn.Module):
    def __init__(self, in_n, out_n):
        super(DeconvBNReLU, self).__init__()
        self.model = nn.Sequential(nn.ConvTranspose2d(
            in_n, out_n, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_n),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.model(x)


class DepthPointConv(nn.Module):
    """深度卷积和逐点卷积"""

    def __init__(self, in_chans, out_chans, stride=1):
        super(DepthPointConv, self).__init__()
        self.conv_bn_relu1 = ConvBnReLU(in_chans, in_chans, stride=stride, groups=in_chans)
        self.conv_bn_relu2 = ConvBnReLU(in_chans, out_chans, kernel_size=1, padding=0)

    def forward(self, x):
        out = self.conv_bn_relu1(x)
        return self.conv_bn_relu2(out)


if __name__ == '__main__':
    import torch

    data = torch.randn((5, 5, 224, 224))
    net = Deconv(5, 3, True)
    print(net(data).shape)
