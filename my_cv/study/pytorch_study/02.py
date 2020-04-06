import torch
import matplotlib.pyplot as plt

# 回归类型的例子
data_shape = torch.ones(400, 2)
x0 = torch.normal(2 * data_shape, 1)
y0 = torch.zeros(data_shape.size()[0])
x1 = torch.normal(-2 * data_shape, 1)
y1 = torch.ones(data_shape.size()[0])

x = torch.cat((x0, x1), 0).type(torch.FloatTensor)
y = torch.cat((y0, y1)).type(torch.LongTensor)
print(y.size())
print(y)


# plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=8, lw=0, cmap='RdYlGn')
# plt.show()


# create network

class Net(torch.nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_input, n_hidden)
        self.output = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = torch.sigmoid(self.hidden(x))
        return self.output(x)


net = Net(n_input=2, n_hidden=10, n_output=2)
print(net)

# train network
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
loss_func = torch.nn.CrossEntropyLoss()

plt.ion()

for i in range(1000):
    out = net(x)

    loss = loss_func(out, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 20 == 0:
        plt.cla()
        # temp = torch.softmax(out, 1)
        prediction = torch.max(out, 1)[1]
        # prediction = torch.max(out)
        pred_y = prediction.data.numpy().squeeze()

        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=8, lw=0, cmap='RdYlGn')
        accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
        plt.text(1.5, -4, 'Accuracy=%.4f' % accuracy, fontdict={'size': 12, 'color': 'orange'})
        plt.pause(0.1)
        if accuracy == 1.0:
            print('perfect')
            break
print('end')
plt.ioff()
plt.show()
