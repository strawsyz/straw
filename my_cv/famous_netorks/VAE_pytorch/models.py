import torch
import torch.nn as nn
from torch.autograd import Variable


class VAE(nn.Module):
    """实现简单的VAE模型"""

    def __init__(self):
        super(VAE, self).__init__()
        # encoder部分
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        # decoder部分
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def reparametrize(self, mu, logvar):
        # 计算std
        std = logvar.mul(0.5).exp_()
        # 按照std的尺寸，创建一个符合正态分布，并转为Variable类型
        eps = Variable(std.data.new(std.size()).normal_())
        # 乘以方差，加上均值，修改分布
        return eps.mul(std).add_(mu)

    def forward(self, x):
        # fc21和fc22一个计算平均值，一个计算方差
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        mean = self.fc21(x)
        var = self.fc22(x)
        # 随机采样
        out = self.reparametrize(mean, var)
        # 输出解码的结合，和输入的x的均值和方差
        out = self.relu(self.fc3(out))
        out = self.fc4(out)
        out = self.sigmoid(out)
        return out, mean, var


# 计算重构误差函数
def loss_function(recon_x, x, mean, var):
    # 将x压平
    x = x.view(-1, 784)
    BCE_loss = nn.BCELoss(reduction='sum')(recon_x, x)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # 计算论文汇总定义的误差函数
    KLD_element = mean.pow(2).add_(var.exp()).mul_(-1).add_(1).add_(var)
    KLD_loss = torch.sum(KLD_element).mul_(-0.5)
    # 将两个函数结合起来
    return BCE_loss + KLD_loss
