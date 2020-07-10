import dgl
import networkx as nx

# create a graph
g_nx = nx.petersen_graph()
g_dgl = dgl.DGLGraph(g_nx)

import matplotlib.pyplot as plt

plt.subplot(121)
nx.draw(g_nx, with_labels=True)
plt.subplot(122)
nx.draw(g_dgl.to_networkx(), with_labels=True)

plt.show()

# add edges and nodes into graph
import dgl
import torch as th

g = dgl.DGLGraph()
g.add_nodes(10)
# A couple edges one-by-one
for i in range(1, 5):
    g.add_edge(i, 0)
# A few more with a paired list
src = list(range(5, 8));
dst = [0] * 3
g.add_edges(src, dst)
# finish with a pair of tensors
src = th.tensor([8, 9]);
dst = th.tensor([0, 0])
g.add_edges(src, dst)
g.add_edges([2], [8])
nx.draw(g.to_networkx(), with_labels=True)
plt.show()
# Edge broadcasting will do star graph in one go!
g.clear();
g.add_nodes(10)
src = th.tensor(list(range(1, 10)));
g.add_edges(src, 0)

import networkx as nx
import matplotlib.pyplot as plt

nx.draw(g.to_networkx(), with_labels=True)
plt.show()

# assigin a feature
import dgl
import torch

# assign node features
x = torch.randn(10, 3)
# g.clear()
g.ndata['x'] = x

# print(g.ndata['x'] == g.nodes[:].data['x'])
print(g.ndata['x'])
print('x value of first node in graph : {}'.format(g.nodes[0].data['x']))
# Access node set with integer, list, or integer tensor
g.nodes[0].data['x'] = torch.zeros(1, 3)
g.nodes[[0, 1, 2]].data['x'] = torch.zeros(3, 3)
g.nodes[torch.tensor([0, 1, 2])].data['x'] = torch.zeros(3, 3)

# Assign edge features
g.edata['w'] = th.randn(9, 2)
print(g.edata['w'])
print('w value of first edge in graph : {}'.format(g.edges[0].data['w']))
# Access edge set with IDs in integer, list, or integer tensor
g.edges[1].data['w'] = th.randn(1, 2)
g.edges[[0, 1, 2]].data['w'] = th.zeros(3, 2)
print("g.edges[[0, 1, 2]].data['w'] : \n{}".format(g.edges[[0, 1, 2]].data['w']))
g.edges[th.tensor([0, 1, 2])].data['w'] = th.zeros(3, 2)

# You can also access the edges by giving endpoints
g.edges[1, 0].data['w'] = th.ones(1, 2)  # edge 1 -> 0
g.edges[[1, 2, 3], [0, 0, 0]].data['w'] = th.ones(3, 2)  # edges [1, 2, 3] -> 0

print(g.node_attr_schemes())
g.ndata['x'] = th.zeros((10, 4))
print(g.node_attr_schemes())

# remove node or edge states
g.ndata.pop('x')
g.edata.pop('w')
print(g.node_attr_schemes())

# create multigraphs
g_multi = dgl.DGLGraph(multigraph=True)
g_multi.add_nodes(10)
g_multi.ndata['x'] = torch.randn(10, 2)
g_multi.add_edges(list(range(1, 10)), 0)
g_multi.add_edge(1, 0)  # two edges on 1->0

g_multi.edata['w'] = th.randn(10, 2)
g_multi.edges[1].data['w'] = th.zeros(1, 2)
print(g_multi.edges())
plt.figure()
nx.draw(g_dgl.to_networkx(), with_labels=True)

plt.show()

# in multigraphs, use edge's id to query edge
eid_10 = g_multi.edge_id(1, 0)
g_multi.edges[eid_10].data['w'] = th.ones(len(eid_10), 2)
print(g_multi.edata['w'])

# !!!!nodes and edges can be added but not remove
