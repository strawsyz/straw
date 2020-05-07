import torch
from torch import nn
from torch.nn import functional as F


# 参考torchvision中的实现

class Inception3(nn.Module):
    def __init__(self, n_classes=1000, aux_logits=True, transform_input=False):
        """

        :param n_classes: 输出的参数
        :param aux_logits: 使用添加辅助输出层
        :param transform_input: 是否需要重新归一化
        """
        super(Inception3, self).__init__()
        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.Conv2d_1a_3x3 = ConvBNReLU(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = ConvBNReLU(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = ConvBNReLU(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1 = ConvBNReLU(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = ConvBNReLU(80, 192, kernel_size=3)
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)
        self.Mixed_6a = InceptionB(288)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)
        if aux_logits:
            self.AuxLogits = InceptionAux(768, n_classes)
        self.Mixed_7a = InceptionD(768)
        self.Mixed_7b = InceptionE(1280)
        self.Mixed_7c = InceptionE(2048)
        self.fc = nn.Linear(2048, n_classes)

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                X = stats.truncnorm(-2, 2, scale=stddev)
                values = torch.Tensor(X.rvs(m.weight.numel()))
                values = values.view(m.weight.size())
                m.weight.data.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.transform_input:
            # 作用是归一化。pytorch的inception v3训练的时候用的均值和标准差为[0.5,0.5,0.5] [0.5,0.5,0.5]。
            # 而之前那些CNN模型的归一化，均值和标准差为[0.229,0.224,0.225] [0.485,0.456,0.406]。
            # 所以这行的语句是将后者的归一化变成前者的归一化。
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        # 299 x 299 x 3
        out = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        out = self.Conv2d_2a_3x3(out)
        # 147 x 147 x 32
        out = self.Conv2d_2b_3x3(out)
        # 147 x 147 x 64
        out = F.max_pool2d(out, kernel_size=3, stride=2)
        # 73 x 73 x 64
        out = self.Conv2d_3b_1x1(out)
        # 73 x 73 x 80
        out = self.Conv2d_4a_3x3(out)
        # 71 x 71 x 192
        out = F.max_pool2d(out, kernel_size=3, stride=2)
        # 35 x 35 x 192
        out = self.Mixed_5b(out)
        # 35 x 35 x 256
        out = self.Mixed_5c(out)
        # 35 x 35 x 288
        out = self.Mixed_5d(out)
        # 35 x 35 x 288
        out = self.Mixed_6a(out)
        # 17 x 17 x 768
        out = self.Mixed_6b(out)
        # 17 x 17 x 768
        out = self.Mixed_6c(out)
        # 17 x 17 x 768
        out = self.Mixed_6d(out)
        # 17 x 17 x 768
        out = self.Mixed_6e(out)
        # 17 x 17 x 768
        if self.training and self.aux_logits:
            # 辅助输出层的结果
            aux = self.AuxLogits(out)
        # 17 x 17 x 768
        out = self.Mixed_7a(out)
        # 8 x 8 x 1280
        out = self.Mixed_7b(out)
        # 8 x 8 x 2048
        out = self.Mixed_7c(out)
        # 8 x 8 x 2048
        out = F.avg_pool2d(out, kernel_size=8)
        # 1 x 1 x 2048
        out = F.dropout(out, training=self.training)
        # 1 x 1 x 2048
        out = out.view(out.size(0), -1)
        # 2048
        out = self.fc(out)
        # 1000 (num_classes)
        if self.training and self.aux_logits:
            # 辅助输出层的结果
            return out, aux
        else:
            return out


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_n, out_n, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.add_module("Conv",
                        nn.Conv2d(in_n, out_n, **kwargs, bias=False))
        self.add_module("BN", nn.BatchNorm2d(out_n, eps=0.001))
        self.add_module("ReLU", nn.ReLU(inplace=True))


class InceptionA(nn.Module):
    def __init__(self, in_n, pool_features):
        """

        :param in_n: 输入的通道数
        :param pool_features: 池化的通道数
        """
        super(InceptionA, self).__init__()

        self.branch1x1 = ConvBNReLU(in_n, 64, kernel_size=1)

        self.branch5x5_1 = ConvBNReLU(in_n, 48, kernel_size=1)
        self.branch5x5_2 = ConvBNReLU(48, 64, kernel_size=5, padding=2)

        self.branch3x3_1 = ConvBNReLU(in_n, 64, kernel_size=1)
        self.branch3x3_2 = ConvBNReLU(64, 96, kernel_size=3, padding=1)
        self.branch3x3_3 = ConvBNReLU(96, 96, kernel_size=3, padding=1)

        # 1x1的卷积，在卷积直线需要先经过池化
        self.branch_pool = ConvBNReLU(in_n, pool_features, kernel_size=1)

    def forward(self, x):
        out1x1 = self.branch1x1(x)

        out5x5 = self.branch5x5_2(self.branch5x5_1(x))

        out3x3 = self.branch3x3_3(self.branch3x3_2(self.branch3x3_1(x)))

        out_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        out_pool = self.branch_pool(out_pool)

        # 将四个输出全都连接起来
        out = [out1x1, out5x5, out3x3, out_pool]
        return torch.cat(out, 1)


class InceptionB(nn.Module):
    def __init__(self, in_channels):
        super(InceptionB, self).__init__()
        self.branch3x3 = ConvBNReLU(in_channels, 384, kernel_size=3, stride=2)

        self.branch3x3db_1 = ConvBNReLU(in_channels, 64, kernel_size=1)
        self.branch3x3db_2 = ConvBNReLU(64, 96, kernel_size=3, padding=1)
        self.branch3x3db_3 = ConvBNReLU(96, 96, kernel_size=3, stride=2)

    def forward(self, x):
        out3x3 = self.branch3x3(x)

        out3x3db = self.branch3x3db_1(x)
        out3x3db = self.branch3x3db_2(out3x3db)
        out3x3db = self.branch3x3db_3(out3x3db)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

        out = [out3x3, out3x3db, branch_pool]
        return torch.cat(out, 1)


class InceptionC(nn.Module):
    def __init__(self, in_channels, channels_7x7):
        super(InceptionC, self).__init__()
        self.branch1x1 = ConvBNReLU(in_channels, 192, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = ConvBNReLU(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = ConvBNReLU(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = ConvBNReLU(c7, 192, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7db_1 = ConvBNReLU(in_channels, c7, kernel_size=1)
        self.branch7x7db_2 = ConvBNReLU(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7db_3 = ConvBNReLU(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7db_4 = ConvBNReLU(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7db_5 = ConvBNReLU(c7, 192, kernel_size=(1, 7), padding=(0, 3))

        self.branch_pool = ConvBNReLU(in_channels, 192, kernel_size=1)

    def forward(self, x):
        out1x1 = self.branch1x1(x)

        out7x7 = self.branch7x7_1(x)
        out7x7 = self.branch7x7_2(out7x7)
        out7x7 = self.branch7x7_3(out7x7)

        out7x7_db = self.branch7x7db_1(x)
        out7x7_db = self.branch7x7db_2(out7x7_db)
        out7x7_db = self.branch7x7db_3(out7x7_db)
        out7x7_db = self.branch7x7db_4(out7x7_db)
        out7x7_db = self.branch7x7db_5(out7x7_db)

        out_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        out_pool = self.branch_pool(out_pool)

        out = [out1x1, out7x7, out7x7_db, out_pool]
        return torch.cat(out, 1)


class InceptionD(nn.Module):
    def __init__(self, in_channels):
        super(InceptionD, self).__init__()
        self.branch3x3_1 = ConvBNReLU(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = ConvBNReLU(192, 320, kernel_size=3, stride=2)

        self.branch7x7x3_1 = ConvBNReLU(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = ConvBNReLU(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = ConvBNReLU(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = ConvBNReLU(192, 192, kernel_size=3, stride=2)

    def forward(self, x):
        out3x3 = self.branch3x3_1(x)
        out3x3 = self.branch3x3_2(out3x3)

        out7x7x3 = self.branch7x7x3_1(x)
        out7x7x3 = self.branch7x7x3_2(out7x7x3)
        out7x7x3 = self.branch7x7x3_3(out7x7x3)
        out7x7x3 = self.branch7x7x3_4(out7x7x3)

        out_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        out = [out3x3, out7x7x3, out_pool]
        return torch.cat(out, 1)


class InceptionE(nn.Module):
    def __init__(self, in_n):
        super(InceptionE, self).__init__()
        self.branch1x1 = ConvBNReLU(in_n, 320, kernel_size=1)

        self.branch3x3_1 = ConvBNReLU(in_n, 384, kernel_size=1)
        self.branch3x3_2a = ConvBNReLU(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = ConvBNReLU(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3_db_1 = ConvBNReLU(in_n, 448, kernel_size=1)
        self.branch3x3_db_2 = ConvBNReLU(448, 384, kernel_size=3, padding=1)
        self.branch3x3_db_3a = ConvBNReLU(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_db_3b = ConvBNReLU(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = ConvBNReLU(in_n, 192, kernel_size=1)

    def forward(self, x):
        out1x1 = self.branch1x1(x)

        out3x3 = self.branch3x3_1(x)
        out3x3 = [
            self.branch3x3_2a(out3x3),
            self.branch3x3_2b(out3x3)
        ]
        out3x3 = torch.cat(out3x3, 1)

        out3x3_db = self.branch3x3_db_1(x)
        out3x3_db = self.branch3x3_db_2(out3x3_db)
        out3x3_db = [
            self.branch3x3_db_3a(out3x3_db),
            self.branch3x3_db_3b(out3x3_db)
        ]
        out3x3_db = torch.cat(out3x3_db, 1)

        out_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        out_pool = self.branch_pool(out_pool)

        out = [out1x1, out3x3, out3x3_db, out_pool]
        return torch.cat(out, 1)


class InceptionAux(nn.Module):
    """
    辅助分类结构
    """

    def __init__(self, in_n, n_classes):
        super(InceptionAux, self).__init__()
        self.conv_1 = ConvBNReLU(in_n, 128, kernel_size=1)
        self.conv_2 = ConvBNReLU(128, 768, kernel_size=5)
        # 设置标准差
        self.conv_2.stddev = 0.01
        self.fc = nn.Linear(768, n_classes)
        self.fc.stddev = 0.001

    def forward(self, x):
        out = F.avg_pool2d(x, kernel_size=5, stride=3)
        out = self.conv_1(out)
        out = self.conv_2(out)
        out = out.view(out.size(0), -1)
        return self.fc(out)


if __name__ == '__main__':
    # 经测试网络可以跑通
    model = Inception3()
    print(model)
    input = torch.rand(4, 3, 299, 299)
    print("size of the output is {}".format(model(input)[0].size()))
    print("size of the aux is {}".format(model(input)[1].size()))
