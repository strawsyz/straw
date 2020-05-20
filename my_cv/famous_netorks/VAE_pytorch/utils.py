import torch


def save_params(net, file_name):
    # 只保存网络中的参数 (速度快, 占内存少)
    torch.save(net.state_dict(), file_name + ".pkl")


def load_params(target_net, path):
    # 将保存的参数复制到 target_net
    target_net.load_state_dict(torch.load(path))
    return target_net
