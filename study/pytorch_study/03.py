import torch


# pytorch中创建网络的方法

class Net_1(torch.nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(Net_1, self).__init__()
        self.hidden = torch.nn.Linear(n_input, n_hidden)
        self.output = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = self.hidden(x)
        x = torch.sigmoid(x)
        return self.output(x)


net_1 = Net_1(1, 10, 1)
print(net_1)

net_2 = torch.nn.Sequential(
    torch.nn.Linear(1, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 1)
)

print(net_2)
