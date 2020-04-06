import torch
import matplotlib.pyplot as plt

n_input = 1
# n_hidden should be very big to make dropout's effect more clear
n_hidden = 100
n_output = 1
EPOCH = 1000
LR = 0.01
torch.manual_seed(1)  # reproducible

N_SAMPLES = 20

# training data
x = torch.unsqueeze(torch.linspace(-1, 1, N_SAMPLES), 1)
y = x + 0.3 * torch.normal(torch.zeros(N_SAMPLES, 1), torch.ones(N_SAMPLES, 1))

# test data
test_x = torch.unsqueeze(torch.linspace(-1, 1, N_SAMPLES), 1)
test_y = test_x + 0.3 * torch.normal(torch.zeros(N_SAMPLES, 1), torch.ones(N_SAMPLES, 1))

# show data
plt.scatter(x.data.numpy(), y.data.numpy(), c='magenta', s=50, alpha=0.5, label='train')
plt.scatter(test_x.data.numpy(), test_y.data.numpy(), c='cyan', s=50, alpha=0.5, label='test')
plt.legend(loc='upper left')
plt.ylim((-2.5, 2.5))
plt.show()

net_overfitting = torch.nn.Sequential(
    torch.nn.Linear(n_input, n_hidden),
    torch.nn.ReLU(),
    torch.nn.Linear(n_hidden, n_hidden),
    torch.nn.ReLU(),
    torch.nn.Linear(n_hidden, n_output)
)

net_dropout = torch.nn.Sequential(
    torch.nn.Linear(n_input, n_hidden),
    torch.nn.Dropout(0.5),
    torch.nn.ReLU(),
    torch.nn.Linear(n_hidden, n_hidden),
    torch.nn.Dropout(0.5),
    torch.nn.ReLU(),
    torch.nn.Linear(n_hidden, n_output)
)

optimizer_overfit = torch.optim.Adam(net_overfitting.parameters(), lr=LR)
optimizer_drop = torch.optim.Adam(net_dropout.parameters(), lr=LR)

loss_func = torch.nn.MSELoss()
plt.ion()
for i in range(EPOCH):
    pred_overfit = net_overfitting(x)
    pred_drop = net_dropout(x)

    loss_overfit = loss_func(pred_overfit, y)
    loss_drop = loss_func(pred_drop, y)

    optimizer_overfit.zero_grad()
    optimizer_drop.zero_grad()

    loss_overfit.backward()
    loss_drop.backward()

    optimizer_overfit.step()
    optimizer_drop.step()

    # 接着上面来
    if i % 10 == 0:     # 每 10 步画一次图
        # change to eval mode in order to fix drop out effect
        net_overfitting.eval()
        # parameters for dropout differ from train mode
        net_dropout.eval()

        # plotting
        plt.cla()
        test_pred_ofit = net_overfitting(test_x)
        test_pred_drop = net_dropout(test_x)
        plt.scatter(x.data.numpy(), y.data.numpy(), c='magenta', s=5, alpha=0.3, label='train')
        plt.scatter(test_x.data.numpy(), test_y.data.numpy(), c='cyan', s=5, alpha=0.3, label='test')
        plt.plot(test_x.data.numpy(), test_pred_ofit.data.numpy(), 'r-', lw=3, label='overfitting')
        plt.plot(test_x.data.numpy(), test_pred_drop.data.numpy(), 'b--', lw=3, label='dropout(50%)')
        plt.text(0, -1.2, 'overfitting loss=%.4f' % loss_func(test_pred_ofit, test_y).data.numpy(),
                 fontdict={'size': 12, 'color': 'red'})
        plt.text(0, -1.5, 'dropout loss=%.4f' % loss_func(test_pred_drop, test_y).data.numpy(),
                 fontdict={'size': 12, 'color': 'orange'})
        plt.legend(loc='upper left');
        plt.ylim((-2.5, 2.5));
        plt.pause(0.1)

        # 将两个网络改回 训练形式
        net_overfitting.train()
        net_dropout.train()
plt.ioff()
plt.show()