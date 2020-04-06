import torch
from matplotlib import pyplot as plt


# 回归问题的例子

# create dataset

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = x.pow(2) + 0.2 * torch.rand(x.size())

# 画图

# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()

# create network
import torch.nn.functional as F


class Net(torch.nn.Module):

    # 定义网络的一些层
    def __init__(self, n_input, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_input, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    # 定义前向传播的方式
    def forward(self, x):
        x = torch.sigmoid(self.hidden(x))
        x = self.predict(x)
        return x


net = Net(n_input=1, n_hidden=10, n_output=1)
# 输出网络结构
print(net)

# train network
# 传入网路的所有参数和决定学习率
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
# 预测值和真实值的误差计算公式
loss_func = torch.nn.MSELoss()

plt.ion()
plt.show()

for i in range(1000):
    # 将数据传入神经网络
    prediction = net(x)
    # 计算真实值和预测结果的误差
    loss = loss_func(prediction, y)
    # pytorch中的backward()函数的计算，当网络参量进行反馈时，梯度是被积累的而不是被替换掉
    optimizer.zero_grad()
    # 误差反向传播，计算参数更新值
    loss.backward()
    # 将参数更新值施加到net的参数上
    optimizer.step()
    # visual training process
    if i % 20 == 0:
        plt.cla()  # 清理之前plot
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), y.data.numpy(), 'r-', lw=1)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 12, 'color': 'orange'})
        plt.pause(0.1)

#　关闭交互模式，最后显示图片
plt.ioff()
plt.show()