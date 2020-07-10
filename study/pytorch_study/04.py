import torch

# 提取保存网络

# create dataset
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = x.pow(2) + 0.2 * torch.rand(x.size())


def save():
    net1 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )
    optimizer = torch.optim.SGD(net1.parameters(), lr=0.01)
    loss_func = torch.nn.MSELoss()

    for i in range(1000):
        prediction = net1(x)
        loss = loss_func(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 保存整个网络
    torch.save(net1, 'net.pkl')
    # 保存网络中的所有参数
    torch.save(net1.state_dict(), 'net_params.pkl')

def save_net(net, file_name):
    # 保存整个网络
    torch.save(net, file_name + ".pkl")


def save_params(net, file_name):
    # 只保存网络中的参数 (速度快, 占内存少)
    torch.save(net.state_dict(), file_name + ".pkl")


def restore_net(file_name):
    return torch.load(file_name + '.pkl')


def restore_params(target_net, file_name):
    # 将保存的参数复制到 target_net
    target_net.load_state_dict(torch.load(file_name + '.pkl'))
    return target_net