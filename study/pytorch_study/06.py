import matplotlib.pyplot as plt
import torch
import torch.utils.data as Data

# 几种优化器进行比较
# SGD, Momentum, RMSprop, Adam

torch.manual_seed(1)

LR = 0.01
BATCH_SIZE = 32
EPOCH = 12

# create dataset
x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
y = x.pow(3) + 0.1 * torch.normal(torch.zeros(*x.size()))

plt.scatter(x.numpy(), y.numpy(), s=9)
plt.show()

torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)


class Net(torch.nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_input, n_hidden)
        self.output = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        return self.output(torch.sigmoid(self.hidden(x)))


LR = 0.01
net_SGD = Net(1, 9, 1)
net_Momentum = Net(1, 9, 1)
net_RMSprop = Net(1, 9, 1)
net_Adam = Net(1, 9, 1)
nets = [net_SGD, net_Momentum, net_RMSprop, net_Adam]

opt_SGD = torch.optim.SGD(net_SGD.parameters(), lr=LR)
opt_Momentum = torch.optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.8)
opt_RMSprop = torch.optim.RMSprop(net_RMSprop.parameters(), lr=LR, alpha=0.9)
opt_Adam = torch.optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))
optimizers = [opt_SGD, opt_Momentum, opt_RMSprop, opt_Adam]

loss_func = torch.nn.MSELoss()
losses_his = [[], [], [], []]

for i in range(EPOCH):
    print('EPOCH:', i)
    for step, (b_x, b_y) in enumerate(loader):
        for net, opt, l_his in zip(nets, optimizers, losses_his):
            output = net(b_x)
            loss = loss_func(output, b_y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            l_his.append(loss.data.numpy())

labels = ['SGD', 'Momentum', 'RMSprop', 'Adam']
for i, l_his in enumerate(losses_his):
    plt.plot(l_his, label=labels[i])
plt.legend(loc='best')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.ylim((0, 0.2))
plt.show()
