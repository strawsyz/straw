import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph

# it can't run in charm by can run in the terminal

# define the message and reduce function
gcn_msg = fn.copy_src(src='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')


# define the node UDF for apply_nodes
class NodeApplyModule(nn.Module):
    def __init__(self, n_input, n_output, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(n_input, n_output)
        self.activation = activation

    def forward(self, node):
        h = self.linear(node.data['h'])
        if self.activation is not None:
            h = self.activation(h)
        return {'h': h}


# define GCN model
class GCN(nn.Module):
    def __init__(self, n_input, n_output, activation):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(n_input, n_output, activation)

    def forward(self, g, feature):
        g.ndata['h'] = feature
        g.update_all(gcn_msg, gcn_reduce)
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop('h')


# a net has two GCN layers
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.gcn1 = GCN(1433, 512, F.sigmoid)
        self.gcn2 = GCN(512, 128, F.sigmoid)
        self.gcn3 = GCN(128, 16, F.sigmoid)
        self.gcn4 = GCN(16, 7, None)

    def forward(self, g, features):
        x = self.gcn1(g, features)
        x = self.gcn2(g, x)
        x = self.gcn3(g, x)
        x = self.gcn4(g, x)
        return x


net = Net()
print(net)

# load cora dataset
from dgl.data import citation_graph as citegrh
import networkx as nx


def load_cora_data():
    data = citegrh.load_cora()
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    train_mask = torch.BoolTensor(data.train_mask)
    test_mask = torch.BoolTensor(data.test_mask)
    g = data.graph
    # add selfloop
    g.remove_edges_from(nx.selfloop_edges(g))
    g = DGLGraph(g)
    g.add_edges(g.nodes(), g.nodes())
    return g, features, labels, train_mask, test_mask


# evaluate function
def evaluate(model, g, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


import time
import numpy as np

# load dataset
g, features, labels, train_mask, test_mask = load_cora_data()
# define optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=0.01)
dur = []
EPOCH = 2001
start_EPOCH = 1
inter = 20
loss_hist = []
acc_hist = []
time_hist = []
epoch_hist = []
for epoch in range(start_EPOCH, EPOCH):
    if epoch >= 3:
        t0 = time.time()
    net.train()
    logits = net(g, features)
    logp = F.log_softmax(logits, 1)
    loss = F.nll_loss(logp[train_mask], labels[train_mask])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch >= 3:
        dur.append(time.time() - t0)
    if epoch % inter == 0:
        acc = evaluate(net, g, features, labels, test_mask)

        loss_hist.append(np.log(loss.item()))
        acc_hist.append(np.log(acc))
        time_hist.append(np.mean(dur))
        epoch_hist.append(epoch)
        print("Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f} | Time(s) {:.4f}".format(
            epoch, loss.item(), acc, np.mean(dur)))

# draw the result
import matplotlib.pyplot as plt

plt.figure()
plt.plot(epoch_hist, loss_hist, label='loss')
plt.plot(epoch_hist, time_hist, label='avgtime')
plt.plot(epoch_hist, acc_hist, label='acc')
# plot(x,y)         # 默认为蓝色实线
# plot(x,y,'go-')   # 带有圆圈标记的绿线
# plot(x,y,'ks:')   # 带有正方形标记的黑色虚线
plt.show()
