import torch.utils.data
from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms

from models import VAE, loss_function
from utils import save_params

BATCH_SIZE = 8
DATASET_PATH = "../data"
lr = 1e-3
# 准备数据
trainset = datasets.MNIST('../data', train=True, download=True, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
# 创建模型
model = VAE()
# 设置优化器
optimizer = optim.Adam(model.parameters(), lr=lr)


def train(n_epoch=10, interval=1000):
    """
    训练用函数
    :param n_epoch: 训练的epoch数
    :param interval: 没interval个batch就打印一次
    :return:
    """
    model.train()
    for epoch in range(n_epoch):
        # 循环每一个epoch
        epoch_loss = 0
        for batch_index, (data, _) in enumerate(train_loader):
            data = Variable(data)
            optimizer.zero_grad()
            # 输入神经网络
            recon_batch, mu, logvar = model(data)
            # 计算损失函数
            loss = loss_function(recon_batch, data, mu, logvar)
            # 反向传播
            loss.backward()
            epoch_loss += loss.data
            optimizer.step()
            if batch_index % interval == 0:
                print('Train Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_index * len(data), len(train_loader.dataset),
                           100. * batch_index / len(train_loader), loss.data / len(data)))
        print("========================")
        print('====> Epoch: {}, average loss : {:.4f}'.format(
            epoch, epoch_loss / len(train_loader.dataset)))
        print("========================")

        # 全部训练完毕之后保存模型
        print("========saving model======")
        save_params(model, 'VAE')
        print("========saved in VAE.pkl=======")


if __name__ == '__main__':
    train()
