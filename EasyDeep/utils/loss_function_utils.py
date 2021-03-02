import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

from utils.data_augm_utils import find_contours


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


from torch.nn import functional as F


class DiceLoss(torch.nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, gt, smooth=1):
        pred = F.sigmoid(pred)
        pred = pred.view(-1)
        gt = gt.view(-1)

        intersection = (pred * gt).sum()
        dice = (2. * intersection + smooth) / pred.sum() + gt.sum() + smooth
        return 1 - dice


class DiceBCELoss(nn.Module):
    def __init__(self):
        super(DiceBCELoss, self).__init__()

    def forward(self, pred, gt, smooth=1):
        pred = F.sigmoid(pred)

        pred = pred.view(-1)
        gt = gt.view(-1)

        intersection = (pred * gt).sum()
        DICE_loss = 1 - (2. * intersection + smooth) / pred.sum() + gt.sum() + smooth
        BCE_loss = F.binary_cross_entropy(pred, gt, reduction="mean")
        return DICE_loss + BCE_loss


class IoULoss(nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()

    def forward(self, pred, gt, smooth=1):
        pred = F.sigmoid(pred)

        pred = pred.view(-1)
        gt = gt.vierw(-1)

        intersection = (pred * gt).sum()
        union = (pred + gt).sum() - intersection
        IoU = (intersection + smooth) / (union + smooth)
        return 1 - IoU


class FocalLoss(nn.Module):
    """https://arxiv.org/abs/1708.02002"""

    def __init__(self, alpha=0.8, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, gt, smooth=1):
        pred = F.sigmoid(pred)

        pred = pred.view(-1)
        gt = gt.view(-1)

        BCE_loss = F.binary_cross_entropy(pred, gt, reduction="mean")
        BCE_exp_loss = torch.exp(BCE_loss)
        return self.alpha * (1 - BCE_exp_loss) ** self.gamma * BCE_loss


class ActivateContourLoss(nn.Module):
    def __init__(self, weight):
        super(ActivateContourLoss, self).__init__()
        self.weight = weight

    def forward(self, pred, gt):
        '''
        y_true, y_pred: tensor of shape (B, C, H, W), where y_true[:,:,region_in_contour] == 1, y_true[:,:,region_out_contour] == 0.
        weight: scalar, length term weight.
        '''
        # length term
        delta_r = pred[:, :, 1:, :] - pred[:, :, :-1, :]  # horizontal gradient (B, C, H-1, W)
        delta_c = pred[:, :, :, 1:] - pred[:, :, :, :-1]  # vertical gradient   (B, C, H,   W-1)

        delta_r = delta_r[:, :, 1:, :-2] ** 2  # (B, C, H-2, W-2)
        delta_c = delta_c[:, :, :-2, 1:] ** 2  # (B, C, H-2, W-2)
        delta_pred = torch.abs(delta_r + delta_c)

        epsilon = 1e-8  # where is a parameter to avoid square root is zero in practice.
        lenth = torch.mean(torch.sqrt(delta_pred + epsilon))  # eq.(11) in the paper, mean is used instead of sum.

        # region term
        C_in = torch.ones_like(pred)
        C_out = torch.zeros_like(pred)

        region_in = torch.mean(pred * (gt - C_in) ** 2)  # equ.(12) in the paper, mean is used instead of sum.
        region_out = torch.mean((1 - pred) * (gt - C_out) ** 2)
        region = region_in + region_out

        loss = self.weight * lenth + region

        return loss
