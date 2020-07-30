import torch
from torch import nn
from torch.autograd import Variable


class CNNAutoEncoder(nn.Module):
    def __init__(self):
        super(CNNAutoEncoder, self).__init__()
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
        )
        self.decoder = nn.Sequential(
            # nn.Sigmoid(),
            nn.Linear(2, 10),
            nn.Sigmoid(),
            nn.Linear(10, 3)
        )

    def forward(self, x):
        encoder_output = self.encoder(x)
        output = self.decoder(encoder_output)
        return output, encoder_output


def train(num_epoch=float("inf"), min_loss=0.00001, max_try_times=10):
    last_loss = float("inf")
    try_times = 0
    epoch = 0
    while True:
        train_loss = train_one_epoch(epoch)
        if train_loss > last_loss:
            try_times += 1
        else:
            try_times = 0
        last_loss = train_loss
        if try_times == max_try_times:
            print("loss don't decrease in {} epoch".format(max_try_times))
            break
        if train_loss < min_loss:
            break
        if num_epoch < epoch:
            break
        epoch += 1
    # save model
    torch.save(net.state_dict(), model_save_path)


def train_one_epoch(epoch):
    train_loss = 0
    net.train()
    global data
    for sample, label in train_data:
        sample = Variable(torch.from_numpy(sample)).double()
        label = Variable(torch.from_numpy(label)).double()
        optimizer.zero_grad()
        out = net(sample)[0]
        loss = loss_function(out, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.data
    train_loss = train_loss / len(train_data)
    print("epoch:{} \t train loss \t {}".format(epoch, train_loss))
    return train_loss


def test():
    all_loss = 0
    net.load_state_dict(torch.load(model_save_path))
    for i, (m, n, (data, gt)) in enumerate(zip(m_data, n_data, test_data)):
        data = Variable(torch.from_numpy(data)).double()
        gt = Variable(torch.from_numpy(gt)).double()
        optimizer.zero_grad()
        out, hidden_ouput = net(data)
        batch_loss = loss_function(out, gt)
        all_loss += batch_loss
        print("m:{},n:{} output of the hidden layer {}".format(m, n, hidden_ouput.data))
    print("train loss \t {}".format(all_loss / len(test_data)))


if __name__ == '__main__':
    import numpy as np

    model_save_path = "AutoEncoder.pkl"
    net = LinearAutoEncoder().double()
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.03, weight_decay=0.0001)

    # prepare data
    train_data = []
    last_loss = 0
    for i in range(100):
        m = np.random.rand() * np.pi
        n = np.random.rand() * np.pi / 2
        x = np.cos(m) * np.sin(n)
        y = np.sin(m) * np.sin(n)
        z = np.cos(n)
        train_data.append((np.array([x, y, z], dtype=np.float32), np.array([x, y, z], dtype=np.float32)))
    test_data = []
    m_data = []
    n_data = []
    for i in range(2000):
        m = np.random.rand() * np.pi
        n = np.random.rand() * np.pi / 2
        m_data.append(m)
        n_data.append(n)
        x = np.cos(m) * np.sin(n)
        y = np.sin(m) * np.sin(n)
        z = np.cos(n)
        test_data.append((np.array([x, y, z]), np.array([x, y, z])))

    # train(min_loss=0.1)

    test()
