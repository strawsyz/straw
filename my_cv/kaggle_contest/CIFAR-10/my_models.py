import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Sequential):
    def __init__(self, in_n, out_n, kernel_size=3, stride=1, padding=1, groups=1, **kwargs):
        super(Conv, self).__init__()
        self.add_module("Conv", nn.Conv2d(in_n, out_n, kernel_size=kernel_size,
                                          stride=stride, padding=padding, groups=groups, bias=False,
                                          **kwargs))


class ConvBN(Conv):
    def __init__(self, in_n, out_n, kernel_size=3, stride=1, padding=1, groups=1, **kwargs):
        super(ConvBN, self).__init__(in_n, out_n, kernel_size, stride, padding, groups=groups, **kwargs)
        self.add_module("BN", nn.BatchNorm2d(out_n))


class ConvBNReLU(ConvBN):
    def __int__(self, in_n, out_n, kernel_size=3, stride=1, padding=1, groups=1, **kwargs):
        super(ConvBNReLU, self).__init__(in_n, out_n, kernel_size, stride, padding, groups=groups ** kwargs)
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


class MobileNetV2(nn.Module):
    def __init__(self, num_classes, bottle=BaseMobileResNet, layers=[1, 2, 3, 4, 3, 3, 1], ratio=[1, 6, 6, 6, 6, 6, 6],
                 strides=[1, 2, 2, 2, 1, 2, 1]):
        self.layers = layers
        self.bottle = bottle
        super(MobileNetV2, self).__init__()
        self.conv_bn = self._conv_bn(3, 32, 2)
        self.layer1 = self._make_layer(32, 16, layers[0], ratio[0], strides[0])
        self.layer2 = self._make_layer(16, 24, layers[1], ratio[1], strides[1])
        self.layer3 = self._make_layer(24, 32, layers[2], ratio[2], strides[2])
        self.layer4 = self._make_layer(32, 64, layers[3], ratio[3], strides[3])
        self.layer5 = self._make_layer(64, 96, layers[4], ratio[4], strides[4])
        self.layer6 = self._make_layer(96, 160, layers[5], ratio[5], strides[5])
        self.layer7 = self._make_layer(160, 320, layers[6], ratio[6], strides[6])
        self.bottom = self._conv1x1(320, 1280)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Sequential(
            nn.Dropout(p=0.8),
            nn.Linear(1280, num_classes),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n) ** .5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv_bn(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.bottom(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # pro = F.softmax(x)
        # return x, pro
        return x

    def _make_layer(self, in_channel, out_channel, layer, ratio, stride):
        cnn = []

        if layer > 1:
            for i in range(layer):
                out_channel = int(out_channel * 1)
                if i == 0:
                    cnn.append(self.bottle(in_channel, out_channel, stride, ratio))
                else:
                    cnn.append(self.bottle(in_channel, out_channel, 1, ratio))
                in_channel = out_channel
        elif layer == 1:
            cnn.append(self.bottle(in_channel, int(out_channel * 1), 1, ratio))
        return nn.Sequential(*cnn)

    def _conv_bn(self, in_channel, out_channel, stride):
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True),
        )

    def _conv1x1(self, in_channel, out_channel, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True),
        )

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


class MobileNetV2_(nn.Module):
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


class HardSwish(nn.Module):
    """h-swish结构"""

    def __init__(self, inplace=True):
        super(HardSwish, self).__init__()
        self.relu6 = nn.ReLU6(inplace)

    def forward(self, x):
        return x * self.relu6(x + 3) / 6


class ConvBNActiv(ConvBN):
    def __init__(self, in_n, out_n, kernel_size=3, stride=1, padding=1, groups=1, activate="relu6", **kwargs):
        """
        卷积 + BN + 激活函数
        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param stride:
        :param groups
        :param activate: 可选relu，relu6，h_swish
        :return:
        """
        # 卷积部分是深度卷积
        super(ConvBNActiv, self).__init__(in_n, out_n, kernel_size, stride, padding, groups=groups, **kwargs)
        if activate == "relu6":
            self.add_module("ReLU6", nn.BatchNorm2d(out_n))
        elif activate == "relu":
            self.add_module("ReLU", nn.ReLU(inplace=True))
        elif activate == "h_swish":
            self.add_module("h_swish", HardSwish())
        else:
            # 如果设置有问题就使用relu
            self.add_module("ReLU", nn.ReLU(inplace=True))


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v



# class SqueezeExcitation(torch.nn.Module):
#     """Squeeze excite block for MobileNet V3."""
#
#     def __init__(self,
#                  in_chans,
#                  out_chans,
#                  divisible_by=8,
#                  squeeze_factor=3,
#                  inner_activation_fn=torch.nn.ReLU(inplace=True),
#                  gating_fn=torch.nn.Sigmoid()):
#         """
#         构建SE模块
#         :param in_chans:
#         :param out_chans:
#         :param divisible_by: 通道数要可以整除这个数字
#         :param squeeze_factor:
#         :param inner_activation_fn:
#         :param gating_fn:
#         """
#         """Constructor.
#
#             squeeze_factor: The factor of squeezing in the inner fully
#                 connected layer.
#             inner_activation_fn: Non-linearity to be used in inner layer.
#             gating_fn: Non-linearity to be used for final gating-function.
#         """
#         super(SqueezeExcitation, self).__init__()
#         squeeze_channels = _make_divisible(in_chans / squeeze_factor,
#                                            divisor=divisible_by)
#
#         layers = [torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)),
#                   torch.nn.Conv2d(in_channels=in_chans,
#                                   out_channels=squeeze_channels,
#                                   kernel_size=1,
#                                   bias=True),
#                   inner_activation_fn,
#                   torch.nn.Conv2d(in_channels=squeeze_channels,
#                                   out_channels=out_chans,
#                                   kernel_size=1,
#                                   bias=True),
#                   gating_fn]
#         self._layers = torch.nn.Sequential(*layers)
#
#     def forward(self, x, squeeze_input_tensor=None):
#         """Forward computation.
#
#         Args:
#             x: A tensor with shape [batch, height, width, depth].
#             squeeze_input_tensor: A custom tensor to use for computing gating
#                 activation. If provided the result will be input_tensor * SE(
#                 squeeze_input_tensor) instead of input_tensor * SE(
#                 input_tensor).
#
#         Returns:
#             Gated input tensor. (e.g. X * SE(X))
#         """
#         if squeeze_input_tensor is None:
#             squeeze_input_tensor = x
#         squeeze_excite = self._layers(squeeze_input_tensor)
#         result = x * squeeze_excite
#         return result


class SqueezeExcitation(nn.Module):
    """todo 要另外一种写法，本来应该是卷积代替全连接"""

    def __init__(self, in_chans, out_chans, kernel_size, divide=4):
        super(SqueezeExcitation, self).__init__()
        mid_channels = in_chans // divide
        self.avgpool = nn.AvgPool2d(kernel_size=kernel_size, stride=1)
        self.se_block = nn.Sequential(
            nn.Linear(in_features=in_chans, out_features=mid_channels),
            nn.ReLU6(inplace=True),
            nn.Linear(in_features=mid_channels, out_features=out_chans),
            HardSwish(inplace=True),
        )

    def forward(self, x):
        b, c, h, w = x.size()
        # 先是平均池化
        out = self.avgpool(x)
        out = out.view(b, -1)
        out = self.se_block(out)
        out = out.view(b, c, 1, 1)
        return out * x


class SEInvertedBottleneck(nn.Module):
    # todo还可以优化
    def __init__(self, in_channels, mid_channels, out_channels, kernel_size, stride, activate, use_se,
                 se_kernel_size):
        super(SEInvertedBottleneck, self).__init__()
        self.stride = stride
        self.use_se = use_se

        self.conv_bn_activ = ConvBNActiv(in_channels, mid_channels, kernel_size=1, padding=0, activate=activate)
        self.depth_conv = ConvBNActiv(mid_channels, mid_channels, kernel_size, stride, padding=(kernel_size - 1) // 2,
                                      groups=mid_channels,
                                      activate=activate)
        if self.use_se:
            self.SEblock = SqueezeExcitation(mid_channels, mid_channels, se_kernel_size)
        # 点卷积
        self.point_conv = ConvBNActiv(mid_channels, out_channels, kernel_size=1, padding=0, activate=activate)
        if self.stride == 1:
            """如果stride是1，就添加残差部分"""
            self.shortcut = ConvBN(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        out = self.depth_conv(self.conv_bn_activ(x))
        if self.use_se:
            out = self.SEblock(out)
        out = self.point_conv(out)
        out = (out + self.shortcut(x)) if self.stride == 1 else out
        return out


class MobileNetV3(nn.Module):
    # 中间层通道数，输出通道数，kernel_size,stride,是否使用SE模块，SE模块的核大小（没有SE模块的话，就设为0）
    large_cfg = [
        (16, 16, 3, 1, 'relu6', False, -1),
        (64, 24, 3, 2, 'relu6', False, -1),
        (72, 24, 3, 1, 'relu6', False, -1),
        (72, 40, 5, 2, 'relu6', True, 28),
        (120, 40, 5, 1, 'relu6', True, 28),
        (120, 40, 5, 1, 'relu6', True, 28),
        (240, 80, 3, 1, 'h_swish', False, -1),
        (200, 80, 3, 1, 'h_swish', False, -1),
        (184, 80, 3, 2, 'h_swish', False, -1),
        (184, 80, 3, 1, 'h_swish', False, -1),
        (480, 112, 3, 1, 'h_swish', True, 14),
        (672, 112, 3, 1, 'h_swish', True, 14),
        (672, 160, 5, 2, 'h_swish', True, 7),
        (960, 160, 5, 1, 'h_swish', True, 7),
        (960, 160, 5, 1, 'h_swish', True, 7),
    ]
    small_cfg = [
        (16, 16, 3, 2, 'relu6', True, 56),
        (72, 24, 3, 2, 'relu6', False, -1),
        (88, 24, 3, 1, 'relu6', False, -1),
        (96, 40, 5, 2, 'h_swish', True, 14),
        (240, 40, 5, 1, 'h_swish', True, 14),
        (240, 40, 5, 1, 'h_swish', True, 14),
        (120, 48, 5, 1, 'h_swish', True, 14),
        (144, 48, 5, 1, 'h_swish', True, 14),
        (288, 96, 5, 2, 'h_swish', True, 7),
        (576, 96, 5, 1, 'h_swish', True, 7),
        (576, 96, 5, 1, 'h_swish', True, 7),
    ]

    def __init__(self, num_classes=1000, type='large'):
        super(MobileNetV3, self).__init__()
        self.type = type

        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            HardSwish(inplace=True),
        )
        self.bottelneck = self._make_layers(type)
        self.last_layers = self._make_last_layers(type)

        self.classifier = nn.Conv2d(in_channels=1280, out_channels=num_classes, kernel_size=1, stride=1)

    def _make_layers(self, type='large'):
        layers = []
        in_chans = 16
        if type == 'large':
            cfg = self.large_cfg
        elif type == "small":
            cfg = self.small_cfg
        else:
            # 暂时设为large模式
            cfg = self.large_cfg
        for mid_chans, out_chans, kernel_size, stride, activ, use_se, se_kernel_size in cfg:
            layers.append(SEInvertedBottleneck(in_chans, mid_chans, out_chans, kernel_size, stride, activ, use_se,
                                               se_kernel_size))
            in_chans = out_chans
        return nn.Sequential(*layers)

    def _make_last_layers(self, type='large'):
        if type == 'large':
            return nn.Sequential(
                ConvBNActiv(160, 960, kernel_size=1, padding=0, activate='h_swish'),
                # nn.Conv2d(in_channels=160, out_channels=960, kernel_size=1, stride=1),
                # nn.BatchNorm2d(960),
                # HardSwish(inplace=True),
                nn.AvgPool2d(kernel_size=7, stride=1),
                nn.Conv2d(in_channels=960, out_channels=1280, kernel_size=1, stride=1),
                HardSwish(inplace=True),
            )
        elif type == 'small':
            return nn.Sequential(
                ConvBNActiv(96, 576, kernel_size=1, padding=0, activate='h_swish'),
                # nn.Conv2d(in_channels=96, out_channels=576, kernel_size=1, stride=1),
                # nn.BatchNorm2d(576),
                # HardSwish(inplace=True),
                nn.AvgPool2d(kernel_size=7, stride=1),
                nn.Conv2d(in_channels=576, out_channels=1280, kernel_size=1, stride=1),
                HardSwish(inplace=True),
            )

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.first_conv(x)
        out = self.bottelneck(out)
        out = self.last_layers(out)
        # 最后使用分类器分类
        out = self.classifier(out)
        return out.view(out.size(0), -1)



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

    model = MobileNetV3(type='small')
    # print(model)
    input = torch.randn(8, 3, 224, 224)
    y = model(input)
    print(y.shape)
