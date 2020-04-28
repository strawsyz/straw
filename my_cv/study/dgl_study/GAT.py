import torch
import torch.nn as nn
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax, GATConv


class GAT(nn.Module):
    def __init__(self, g, num_layers, in_dim, num_hidden, num_classes, heads,
                 activation, feat_drop, attn_drop, negative_slope, residual):
        super(GAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.gat_layers.append(GATConv(in_dim, num_hidden, heads[0]),
                               feat_drop, attn_drop, negative_slope, False, self.activation)

        for i in range(1, num_layers):
            self.gat_layers.append(GATConv(
                num_hidden * heads[i - 1], num_hidden, heads[i],
                feat_drop, attn_drop, negative_slope, residual, self.activation
            ))
        self.gat_layers.append(GATConv(num_hidden * heads[-2], num_classes, heads[-1],
                                       feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](self.g, h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](self.g, h).mean(1)
        return logits