import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import torchvision.datasets as datasets

# autoencoder


EPOCH = 10
BATCH_SIZE = 64
LR = 0.005
DOWNLOAD_MNIST = True
N_TEST_IMG = 5

# Mnist digits dataset
train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,  # this is training data
    transform=torchvision.transforms.ToTensor(),  # Converts a PIL.Image or numpy.ndarray to
    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=DOWNLOAD_MNIST,  # download it if you don't have it
)
test_data = datasets.MNIST(root='./mnist/', train=False)

# 批训练 50samples, 1 channel, 28x28 (50, 1, 28, 28)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# 为了节约时间, 我们测试时只测试前2000个
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[
         :2000] / 255.  # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_y = test_data.test_labels[:2000]


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 12),
            nn.Tanh(),
            nn.Linear(12, 3)  # encoder to three features which are used to plot 3D image
        )

        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.Tanh(),
            nn.Linear(12, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid()  # make the output's value between 0 and 1
        )

    def forward(self, x):
        encoder = self.encoder(x)
        decoder = self.decoder(encoder)
        return encoder, decoder


auto_encoder = AutoEncoder()

optimizer = torch.optim.Adam(auto_encoder.parameters(), lr=LR)
loss_func = nn.MSELoss()

for epoch in range(EPOCH):
    for step, (x, b_label) in enumerate(train_loader):
        y_b = x_b = x.view(-1, 28 * 28)  # shape is (batch, 28*28)
        # y_b = x.view(-1, 28 * 28)

        encoded, decoded = auto_encoder(x_b)

        loss = loss_func(decoded, y_b)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# visual the result of the training
view_data = train_data.train_data[:200].view(-1, 28 * 28).type(torch.FloatTensor) / 255
encoded_data, _ = auto_encoder(view_data)
from matplotlib import pyplot as plt, cm

fig = plt.figure(2)
from mpl_toolkits.mplot3d import Axes3D

ax = Axes3D(fig)
X = encoded_data.data[:, 0].numpy()
Y = encoded_data.data[:, 1].numpy()
Z = encoded_data.data[:, 2].numpy()

values = train_data.train_labels[:200].numpy()
for x, y, z, s in zip(X, Y, Z, values):
    c = cm.rainbow(int(255 * s / 9))
    ax.text(x, y, z, s, backgroundcolor=c)

ax.set_xlim(X.min(), X.max())
ax.set_ylim(Y.min(), Y.max())
ax.set_zlim(Z.min(), Z.max())

plt.show()