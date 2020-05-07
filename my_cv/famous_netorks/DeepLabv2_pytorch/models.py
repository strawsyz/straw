import torch
from torch import nn
import torch.nn.functional as F

# 参考https://zhuanlan.zhihu.com/p/68531147的实现

class ASPP(nn.Module):
    """ASSP 的实现"""
    def __init__(self, in_n, out_n, rates):
        super(ASPP, self).__init__()

        for i, rate in enumerate(rates):
            # 添加空洞卷积
            # 为了保证输出都是大小一样的
            # padding和dilation的值设为一样的
            self.add_module("Conv{}".format(i),
                            nn.Conv2d(in_n, out_n, 3, 1,
                                      padding=rate, dilation=rate, bias=True))

        # 初始化网络参数
        for m in self.children():
            # 初始化权重的默认值
            nn.init.normal_(m.weight, mean=0, std=0.01)
            # 初始化bias的默认值
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 将每一层的输出加起来然后返回
        return sum([stage(x) for stage in self.children()])


class DeepLabv2(nn.Sequential):
    def __init__(self, n_classes, n_blocks, atrous_rates):
        super(DeepLabv2, self).__init__()
        ch = [64 * 2 ** i for i in range(6)]
        self.add_module("Layer1", Stem(ch[0]))
        self.add_module("Layer2", ResLayer(n_blocks[0], ch[0], ch[2], 1, 1))
        self.add_module("Layer3", ResLayer(n_blocks[1], ch[2], ch[3], 2, 1))
        self.add_module("Layer4", ResLayer(n_blocks[2], ch[3], ch[4], 1, 2))
        self.add_module("Layer5", ResLayer(n_blocks[3], ch[4], ch[5], 1, 4))
        # 添加ASSP模块
        self.add_module("Aspp", ASPP(ch[5], n_classes, atrous_rates))

    def freeze_bn(self):
        """冻结BN层的参数"""
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class ConvBnReLU(nn.Sequential):
    """卷积层+BN+ReLU"""

    def __init__(self, in_n, out_n, kernel_size, stride, padding,
                 dilation, relu=True):
        super(ConvBnReLU, self).__init__()
        # 添加卷积层
        self.add_module("Conv",
                        nn.Conv2d(in_n, out_n, kernel_size,
                                  stride, padding, dilation, bias=False))
        # 添加BN层
        self.add_module("Bn", nn.BatchNorm2d(out_n, eps=1e-5, momentum=0.999))
        if relu:
            # 如果需要使用relu就添加relu层
            self.add_module("ReLU", nn.ReLU())


class Bottleneck(nn.Module):
    def __init__(self, in_n, out_n, stride, dilation, downsample):
        super(Bottleneck, self).__init__()
        # 不明白
        BOTTLENECK_EXPANSION = 4
        mid_n = out_n // BOTTLENECK_EXPANSION
        self.reduce = ConvBnReLU(in_n, mid_n, 1, stride, 0, 1, True)

        self.conv3x3 = ConvBnReLU(mid_n, mid_n, 3, 1, padding=dilation,
                                  dilation=dilation, relu=True)

        self.increase = ConvBnReLU(mid_n, out_n, 1, 1, 0, 1, False)
        self.shortcut = (
            ConvBnReLU(in_n, out_n, 1, stride, 0, 1, False)
            # 如果需要下采样层，就加入，否则直接输出
            if downsample
            else lambda x: x
        )

    def forward(self, x):
        out = self.reduce(x)
        out = self.conv3x3(out)
        out = self.increase(out)
        # 最后将shortcut连接过来
        out += self.shortcut(x)
        # 最后加入relu激活层
        return F.relu(out)


class ResLayer(nn.Sequential):
    def __init__(self, n_layers, in_n, out_n, stride, dilation, dilations_rates=None):
        # todo 入参可以再修改一下
        super(ResLayer, self).__init__()

        if dilations_rates is None:
            dilations_rates = [1 for _ in range(n_layers)]
        else:
            # 保证输入数据，没有问题
            assert n_layers == len(dilations_rates)

        for i in range(1, n_layers + 1):
            self.add_module(
                "Block{}".format(i),
                # 如果是第一层，输入通道是in_n
                Bottleneck(
                    in_n=(in_n if i == 1 else out_n),
                    out_n=out_n,
                    stride=(stride if i == 1 else 1),
                    dilation=dilation * dilations_rates[i-1],
                    downsample=(True if i == 1 else False)
                )

            )


class Stem(nn.Sequential):
    def __init__(self, out_n):
        super(Stem, self).__init__()
        # 固定输入3通道图像
        self.add_module("Conv1", ConvBnReLU(3, out_n, 7, 2, 3, 1))
        self.add_module("Pool", nn.MaxPool2d(3, 2, 1, ceil_mode=True))


if __name__ == '__main__':
    ## 经测试，可以跑通，有些变量名还可以优化

    model = DeepLabv2(
        n_classes=21, n_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18, 24]
    )
    model.eval()
    image = torch.randn(1, 3, 513, 513)

    # print(model)
    print("input:", image.shape)
    print("output:", model(image).shape)
