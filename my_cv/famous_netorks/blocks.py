from torch import nn


class Conv(nn.Sequential):
    """自定义的卷积层，默认核的大小是3，偏移1，padding是1"""

    def __init__(self, in_n, out_n, kernel_size=3, stride=1, padding=1, **kwargs):
        super(Conv, self).__init__()
        # 目前还没有貌似没有看到需要bias的卷积层，就全部设为False
        self.add_module("Conv",
                        nn.Conv2d(in_n, out_n, kernel_size=kernel_size,
                                  stride=stride, padding=padding, bias=False,
                                  **kwargs))


class ConvBN(Conv):
    # 自定义的卷积 + 普通的BN
    def __init__(self, in_n, out_n, kernel_size=3, stride=1, padding=1, **kwargs):
        super(ConvBN, self).__init__(in_n, out_n, kernel_size, stride, padding, **kwargs)
        self.add_module("BN", nn.BatchNorm2d(out_n))


class ConvBNReLU(ConvBN):
    def __int__(self, in_n, out_n, kernel_size=3, stride=1, padding=1, **kwargs):
        super(ConvBNReLU, self).__init__(in_n, out_n, kernel_size, stride, padding, **kwargs)
        self.add_module("ReLU", nn.ReLU(inplace=True))


class DepthPointConv(nn.Module):
    """深度卷积和逐点卷积"""

    def __init__(self, in_chans, out_chans, stride=1):
        super(DepthPointConv, self).__init__()
        self.conv_bn_relu1 = ConvBNReLU(in_chans, in_chans, stride=stride, groups=in_chans)
        self.conv_bn_relu2 = ConvBNReLU(in_chans, out_chans, kernel_size=1, padding=0)

    def forward(self, x):
        out = self.conv_bn_relu1(x)
        return self.conv_bn_relu2(out)
