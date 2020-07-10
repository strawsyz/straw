import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

# use rnn to solve regression problem


torch.manual_seed(1)  # reproducible

# Hyper Parameters
TIME_STEP = 10  # rnn time step / image height
INPUT_SIZE = 1  # rnn input size / image width
LR = 0.02  # learning rate
DOWNLOAD_MNIST = False  # set to True if haven't download the data


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.RNN(  # 这回一个普通的 RNN 就能胜任
            input_size=1,
            hidden_size=32,  # rnn hidden unit
            num_layers=1,  # 有几层 RNN layers
            batch_first=True,  # input & output 会是以 batch size 为第一维度的特征集 e.g. (batch, time_step, input_size)
        )
        self.out = nn.Linear(32, 1)

    def forward(self, x, h_state):  # 因为 hidden state 是连续的, 所以我们要一直传递这一个 state
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, output_size)
        r_out, h_state = self.rnn(x, h_state)  # h_state 也要作为 RNN 的一个输入

        # first way,this way is used to plot the process of the training
        outs = []  # 保存所有时间点的预测值
        for time_step in range(r_out.size(1)):  # 对每一个时间点计算 output
            outs.append(self.out(r_out[:, time_step, :]))
        return torch.stack(outs, dim=1), h_state
        # second way,in this way, quantity of computaion is less
        r_out = r_out.view(-1, 32)
        outs = self.out(r_out)
        return outs.views(-1, 32, TIME_STEP), h_state


rnn = RNN()
print(rnn)
"""
RNN (
  (rnn): RNN(1, 32, batch_first=True)
  (out): Linear (32 -> 1)
)
"""
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_func = nn.MSELoss()

h_state = None
plt.figure(1, figsize=(24, 5))
plt.ion()  # continuously plot
for step in range(300):
    start, end = step * np.pi, (step + 1) * np.pi
    steps = np.linspace(start, end, 10, dtype=np.float32)
    x_np = np.sin(steps)
    y_np = np.cos(steps)
    # np.newaxis　is None's alias
    # shape of x is (batch,time_step,input_size)
    x = torch.from_numpy(x_np[np.newaxis, :, np.newaxis])
    y = torch.from_numpy(y_np[np.newaxis, :, np.newaxis])

    prediction, h_state = rnn(x, h_state)

    h_state = h_state.data

    loss = loss_func(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # plotting
    plt.plot(steps, y_np.flatten(), 'r-')
    plt.plot(steps, prediction.data.numpy().flatten(), 'b-')
    plt.draw();
    plt.pause(0.05)
plt.ioff()
plt.show()
