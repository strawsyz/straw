import dgl
import matplotlib.pyplot as plt
import networkx as nx
import torch

# In the message passing and feature transformations are user-defined functions(UDFs)

N = 100  # number of nodes
DAMP = 0.85  # damping factor 阻尼系数
K = 10  # number of iterations
g = nx.nx.erdos_renyi_graph(N, 0.1)
g = dgl.DGLGraph(g)
nx.draw(g.to_networkx(), node_size=50, node_color=[[.5, .5, .5, ]])
plt.show()

# initialize the pagerank value of each node to 1/n
# and then store each node's out-degree as a node feature
g.ndata['pv'] = torch.ones(N) / N
g.ndata['deg'] = g.out_degrees(g.nodes()).float()


# define the message function
# divides every node's PageRank by its out-degree and passes the result
# as message to its neighbors
# Edge UDFs
def pagerank_message_func(edges):
    return {'pv': edges.src['pv'] / edges.src['deg']}


# define the reduce function
# removes and aggregates the messages from its mailbox
# and computes its new PageRank value
# Node UDFs
# arguments nodes
# return data and mailbox
# data : node features
# mailbox : contains all incoming message features
def pagerank_reduce_func(nodes):
    msgs = torch.sum(nodes.mailbox['pv'], dim=1)
    pv = (1 - DAMP) / N + DAMP * msgs
    return {'pv': pv}


# register the message function and reduce function, which will be called later by DGL
g.register_message_func(pagerank_message_func)
g.register_reduce_func(pagerank_reduce_func)


# the code for PageRank iteration
def pagerank_naive(g):
    for u, v in zip(*g.edges()):
        g.send((u, v))
    for u in g.nodes():
        g.recv(v)


# trigger message and reduce functions on multiple nodes and edges at one time
def pagerank_batch(g):
    g.send(g.edges())
    g.recv(g.nodes())


# use higher-level APIs for efficiency
def pagerank_level2(g):
    g.update_all()


# builtin functions in efficiency
import dgl.function as fn

# directly provide the UDFs to the update_all as its arguments.
# This will override the previously registered UDFs.
def pagerank_builtin(g):
    g.ndata['pv'] = g.ndata['pv'] / g.ndata['deg']
    g.update_all(message_func=fn.copy_src(src='pv', out='m'),
                 reduce_func=fn.sum(msg='m', out='m_sum'))
    g.ndata['pv'] = (1 - DAMP) / N + DAMP * g.ndata['m_sum']


for k in range(K):
    # Uncomment the corresponding line to select different version.
    # pagerank_naive(g)
    # pagerank_batch(g)
    pagerank_level2(g)
    # pagerank_builtin(g)
print(g.ndata['pv'])
# tensor([0.0074, 0.0102, 0.0090, 0.0100, 0.0116, 0.0108, 0.0100, 0.0133, 0.0097,
#         0.0065, 0.0098, 0.0105, 0.0065, 0.0090, 0.0099, 0.0081, 0.0148, 0.0142,
#         0.0138, 0.0050, 0.0090, 0.0074, 0.0090, 0.0095, 0.0090, 0.0115, 0.0092,
#         0.0090, 0.0107, 0.0119, 0.0115, 0.0084, 0.0092, 0.0116, 0.0074, 0.0082,
#         0.0100, 0.0110, 0.0131, 0.0072, 0.0106, 0.0091, 0.0128, 0.0124, 0.0123,
#         0.0123, 0.0117, 0.0085, 0.0078, 0.0116, 0.0100, 0.0101, 0.0100, 0.0139,
#         0.0107, 0.0099, 0.0042, 0.0082, 0.0138, 0.0132, 0.0106, 0.0082, 0.0077,
#         0.0058, 0.0116, 0.0083, 0.0074, 0.0100, 0.0083, 0.0140, 0.0083, 0.0100,
#         0.0090, 0.0074, 0.0118, 0.0075, 0.0140, 0.0091, 0.0099, 0.0127, 0.0090,
#         0.0093, 0.0073, 0.0117, 0.0123, 0.0074, 0.0093, 0.0131, 0.0099, 0.0092,
#         0.0168, 0.0099, 0.0107, 0.0042, 0.0106, 0.0081, 0.0101, 0.0110, 0.0090,
#         0.0121])