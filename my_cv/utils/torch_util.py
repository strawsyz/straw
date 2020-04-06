import torch


def save_net(net, file_name):
    # 保存整个网络
    torch.save(net, file_name + ".pkl")


def save_params(net, file_name):
    # 只保存网络中的参数 (速度快, 占内存少)
    torch.save(net.state_dict(), file_name + ".pkl")


def restore_net(file_name):
    return torch.load(file_name + '.pkl')


def restore_params(target_net, path):
    # 将保存的参数复制到 target_net
    target_net.load_state_dict(torch.load(path))
    return target_net

