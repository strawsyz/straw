# -*- coding: utf-8 -*-
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataset import NYU
from models import FCN
from PIL import Image

# 设置训练参数
BATCH_SIZE = 2
# 数据的路径
DATA_PATH = ""
# 模型保存的路径
MODEL_PATH = ""
# 保存预测结果的文件夹,需要以/结尾
SAVE_PATH = ""

test_data = DataLoader(NYU(DATA_PATH), batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
net = FCN().cuda()
# 变成验证模式
net.eval()
# 加载模型
net.load_state_dict(net.load(MODEL_PATH))

print("===============start testing================")
for i, (data, label) in enumerate(test_data):
    # 将数据放入GPU
    data = Variable(data).cuda()
    label = Variable(label).cuda()
    # 用预测结果
    out = F.log_softmax(net(data), dim=1)
    predict = out.max(1)[1].squeeze().cpu().data().numpy()
    predict = Image.fromarray(predict)
    # 结果
    predict.save("{}{}.png".format(SAVE_PATH, str(i)))

print("================testing end=================")
