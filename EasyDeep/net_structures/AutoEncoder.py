from torch.autograd import Variable
import torch
from torch import nn


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class LinearAutoEncoder(nn.Module):
    def __init__(self):
        super(LinearAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(3, 10),
            nn.Sigmoid(),
            nn.Linear(10, 2),
            nn.Sigmoid(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 10),
            nn.Sigmoid(),
            nn.Linear(10, 3)
        )

    def forward(self, x):
        output = self.encoder(x)
        output = self.decoder(output)
        return output


def train(num_epoch):
    for epoch in range(num_epoch):
        train_one_epoch(epoch)


def train_one_epoch(epoch):
    train_loss = 0
    net.train()
    global data
    for sample, label in train_data:
        sample = Variable(torch.from_numpy(sample)).double()
        label = Variable(torch.from_numpy(label)).double()
        optimizer.zero_grad()
        out = net(sample)
        loss = loss_function(out, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.data
    train_loss = train_loss / len(train_data)
    print("train loss \t {}".format(train_loss))


def test():
    all_loss = 0
    for i, (data, gt) in enumerate(test_data):
        data = Variable(torch.from_numpy(data)).double()
        gt = Variable(torch.from_numpy(gt)).double()
        batch_loss = test_one_batch(data, gt)
        all_loss += batch_loss
    print("train loss \t {}".format(all_loss / 20))


def test_one_batch(data, gt):
    optimizer.zero_grad()
    out = net(data)
    global loss_function
    loss = loss_function(out, gt)
    return loss


if __name__ == '__main__':
    import numpy as np

    net = LinearAutoEncoder().double()
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.0003, weight_decay=0.0001)
    train_data = []
    for i in range(80):
        m = np.random.rand() * np.pi
        n = np.random.rand() * np.pi / 2
        x = np.cos(m) * np.sin(n)
        y = np.sin(m) * np.sin(n)
        z = np.cos(n)
        train_data.append((np.array([x, y, z], dtype=np.float32), np.array([x, y, z], dtype=np.float32)))
    test_data = []
    for i in range(20):
        m = np.random.rand() * np.pi
        n = np.random.rand() * np.pi / 2
        x = np.cos(m) * np.sin(n)
        y = np.sin(m) * np.sin(n)
        z = np.cos(n)
        test_data.append((np.array([x, y, z]), np.array([x, y, z])))

    train(100)

    test()
