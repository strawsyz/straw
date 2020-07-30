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
    def __init__(self, in_n, out_n, kernel_size=3, stride=1, padding=1, bias=False):
        super(Conv, self).__init__()
        self.add_module("Conv2d",
                        nn.Conv2d(in_n, out_n, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))


class ConvReLU(Conv):
    def __init__(self, in_n, out_n, kernel_size=3, stride=1, padding=1, bias=False):
        super(ConvReLU, self).__init__(in_n, out_n, kernel_size, stride, padding, bias)
        self.add_module("ReLU", nn.ReLU(inplace=True))


class LinearReLU(nn.Sequential):
    def __init__(self, in_n, out_n):
        super(LinearReLU, self).__init__()
        self.add_module("Linear", nn.Linear(in_n, out_n))
        self.add_module("ReLU", nn.ReLU(inplace=True))