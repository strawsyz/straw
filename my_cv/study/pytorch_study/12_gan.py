import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# gan

torch.manual_seed(1)
np.random.seed(1)

BATCH_SIZE = 64

LR_G = 0.0003  # learning rate of generator
LR_D = 0.0003  # learning rate of discriminator
N_IDEAS = 5  # the number of ideas for generating an art work
ART_COMPONENTS = 15  # total point G can draw in the canvas
PAINT_POINTS = np.vstack([np.linspace(-1, 1, ART_COMPONENTS) for i in range(BATCH_SIZE)])


def artist_works():
    a = np.random.uniform(1, 2, size=BATCH_SIZE)[:, np.newaxis]
    paintings = a * np.power(PAINT_POINTS, 2) + (a - 1)
    paintings = torch.from_numpy(paintings).float()
    return paintings


G = nn.Sequential(
    nn.Linear(N_IDEAS, 128),
    nn.ReLU(),
    nn.Linear(128, ART_COMPONENTS)
)
D = nn.Sequential(
    nn.Linear(ART_COMPONENTS, 128),
    nn.ReLU(),
    nn.Linear(128, 1),
    nn.Sigmoid()
)

opt_D = torch.optim.Adam(D.parameters(), lr=LR_D)
opt_G = torch.optim.Adam(G.parameters(), lr=LR_G)

plt.ion()

for step in range(10000):
    artist_paintings = artist_works()
    G_ideas = torch.randn(BATCH_SIZE, N_IDEAS)
    G_paintings = G(G_ideas)  # generator create the new paintings

    # D net try to increase prob_artist0,and decrease prob_artist1
    # G net try to increase prob_artist1
    prob_artist0 = D(artist_paintings)
    prob_artist1 = D(G_paintings)

    D_loss = -torch.mean(torch.log(prob_artist0) + torch.log(1. - prob_artist1))
    G_loss = torch.mean(torch.log(1. - prob_artist1))

    opt_D.zero_grad()
    D_loss.backward(retain_graph=True)  # reusing computational graph
    opt_D.step()

    opt_G.zero_grad()
    G_loss.backward()
    opt_G.step()

    if step % 50 == 0:
        plt.cla()
        plt.plot(PAINT_POINTS[0], G_paintings.data.numpy()[0], c='orange', lw=3, label='Generated painting')
        plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='red', lw=3, label='upper bound')
        plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='red', lw=3, label='lower bound')
        plt.text(-5, 2.3, 'D accuracy=%.2f(0.5 for D to converge)' % prob_artist0.data.numpy().mean(),
                 fontdict={'size': 13})
        plt.text(-5, 2, 'D score=%.2f(-1.38 for G to converge)' % -D_loss.data.numpy(), fontdict={'size': 13})
        plt.ylim((0, 3))
        plt.legend(loc='upper right', fontsize=10)
        plt.draw()
        plt.pause(0.1)

plt.ioff()
plt.show()
