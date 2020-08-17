from torch.autograd import Variable
from torch import nn
from utils.data_augm_utils import find_contours


class CenterLoss(nn.Module):
    """找到mask部分的边界部分，
    如果和mask的中心部分重合，能获得更多的分数"""

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


if __name__ == '__main__':
    image_path = ""
    from PIL import Image
    import numpy as np

    image = np.array(Image.open(image_path))


    # 创建图像的轮廓图像
    def edge_detector(source_path, save_path):
        import cv2
        img = cv2.imread(source_path)
        img = cv2.resize(img, (224, 224))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # canny_gray(gray)
        # detected_edges = cv2.GaussianBlur(gray, (3, 3), 0)
        detected_edges = cv2.Canny(gray, 60, 110, apertureSize=3)
        # just add some colours to edges from original image.
        dst = cv2.bitwise_and(img, img, mask=detected_edges)
        # cv2.imshow('canny demo', dst)
        # print(type(dst))
        cv2.imwrite(save_path, dst)
