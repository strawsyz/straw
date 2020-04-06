import time
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.output = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.output(x)
        return x


def draw_loss_history(loss_history):
    plt.figure()
    epoch = len(loss_history)
    plt.plot(np.linspace(1, epoch, epoch, endpoint=True), loss_history)
    plt.show()


if __name__ == '__main__':
    start = time.time()

    x = torch.unsqueeze(torch.linspace(0, 4 * np.pi, 31), dim=1)
    # X = np.linspace(0, 4 * np.pi, 101, endpoint=True)
    # y = 2*x
    y = 10 * np.exp(-0.4 * x) * np.sin(x)

    x, y = Variable(x), Variable(y)
    net = Net(1, 121, 1)
    print(net)

    optimizer = torch.optim.SGD(net.parameters(), lr=0.003)
    loss_function = torch.nn.MSELoss()

    plt.ion()
    plt.show()

    # for t in range(300):
    result_loss = np.inf
    epoch = 0
    loss_history = []
    while result_loss > 0.01:
        epoch = epoch + 1
        prediction = net(x)
        loss = loss_function(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        result_loss = loss.data.numpy()
        loss_history.append(loss.data.numpy())
        # if epoch % 10 == 0:
        #     plt.cla()
        #     plt.scatter(x.data.numpy(), y.data.numpy())
        #     plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=2)
        #     plt.text(0.5, 0, 'L=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        # plt.pause(0.1)
        print('epoch is', epoch)
        print(": loss ", loss.data.numpy())

    for name, param in net.named_parameters():
        print(name, '      ', param)
    print(net.output)
    # 画出训练过程中loss的变化过程
    log_loss = np.log10(loss_history)
    draw_loss_history(log_loss)

    plt.figure()
    pred_Y = net(x)
    plt.plot(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), pred_Y.data.numpy())
    plt.show()

    end = time.time()
    print(str(int(end - start)))
    print('秒')
