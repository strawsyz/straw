from torch.autograd import Variable
from torch import nn
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


