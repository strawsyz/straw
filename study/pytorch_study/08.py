import torch
import torch.utils.data as Data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import nn

# 使用ｒｎｎ来解决分类问题
torch.manual_seed(1)
# Hyper Parameters
EPOCH = 2
BATCH_SIZE = 64
TIME_STEP = 28
INPUT_SIZE = 28
LR = 0.01
DOWNLOAD_MNIST = True
# Mnist 手写数字
train_data = datasets.MNIST(
    root='./mnist/',  # 保存或者提取位置
    train=True,  # this is training data
    transform=transforms.ToTensor(),  # 转换 PIL.Image or numpy.ndarray 成
    # torch.FloatTensor (C x H x W), 训练的时候 normalize 成 [0.0, 1.0] 区间
    download=DOWNLOAD_MNIST,  # 没下载就下载, 下载了就不用再下了
)
test_data = datasets.MNIST(root='./mnist/', train=False)

# 批训练 50samples, 1 channel, 28x28 (50, 1, 28, 28)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# 为了节约时间, 我们测试时只测试前2000个
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[
         :2000] / 255.  # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_y = test_data.test_labels[:2000]


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(  # use LSTM
            input_size=28,
            hidden_size=64,  # RNN hidden unit
            num_layers=1,  # the number of layers
            batch_first=True  # input and output use batch_size as the first dim e.g.(batch,time_step,input_size)
        )

        self.output = nn.Linear(64, 10)  # output layer

    def forward(self, x):
        # 下面的None表示，hidden state　会用全０的state
        r_out, (h_n, h_c) = self.rnn(x, None)

        out = self.output(r_out[:, -1, :])
        return out


rnn = RNN()
print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (x, b_y) in enumerate(train_loader):
        b_x = x.view(-1, 28, 28)  # reshape x to (batch, time_step, input_size)
        output = rnn(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


test_output = rnn(test_x[:10].view(-1, 28, 28))
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(pred_y, 'prediction number')
print(test_y[:10], 'true number')