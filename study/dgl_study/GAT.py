import torch.nn as nn
from dgl.nn.pytorch import GATConv


class GAT(nn.Module):
    def __init__(self, g, num_hidden, heads, n_layers=10, in_dim=500, num_classes=3,
                 activation="relu", feat_drop=0, attn_drop=0, negative_slope=0.7, residual=True):
        super(GAT, self).__init__()
        self.g = g
        self.n_layers = n_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.gat_layers.append(GATConv(in_dim, num_hidden, heads[0]),
                               feat_drop, attn_drop, negative_slope, False, self.activation)

        for i in range(1, n_layers):
            self.gat_layers.append(
                # heads决定输入数据的大小
                GATConv(num_hidden * heads[i - 1], num_hidden, heads[i],
                        feat_drop, attn_drop, negative_slope, residual, self.activation))

        self.gat_layers.append(GATConv(num_hidden * heads[-2], num_classes, heads[-1],
                                       feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, inputs):
        h = inputs
        for layer in range(self.n_layers):
            h = self.gat_layers[layer](self.g, h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](self.g, h).mean(1)
        return logits


if __name__ == '__main__':
    GAT()
