from torch.autograd import Variable
from torch import nn
from utils.data_augm_utils import find_contours
import numpy as np
import torch


class CenterLoss(nn.Module):
    """Center area have high weight and border of mask have low wight"""

    def __init__(self, deta):
        super(CenterLoss, self).__init__()
        self.deta = deta

    def forward(self, output, gt, source_path):
        loss = nn.NLLLoss2d(output, gt)
        weights = find_contours(source_path)
        np.where(weights == 127, self.deta, 1)
        loss = loss * weights
        loss = Variable(loss, requires_grad=True)
        return loss


class FReLU(nn.Module):
    r""" FReLU formulation. The funnel condition has a window size of kxk. (k=3 by default)
        https://arxiv.org/pdf/2007.11824.pdf
    """

    def __init__(self, in_channel, k=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, in_channel, k, padding=1, groups=in_channel, bias=False)
        self.bn = nn.BatchNorm2d(in_channel)

    def forward(self, x):
        return torch.max(x, self.bn(self.conv(x)))
