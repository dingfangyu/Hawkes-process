import matplotlib.pyplot as plt 
import numpy as np 
import networkx as nx 

a = np.load("./a.npy")
u = np.load("./u.npy")
b = np.load("./b.npy")

G = nx.DiGraph()

for i in range(a.shape[0]):
    for j in range(a.shape[1]):
        if a[i][j] > 0.07 and i != j:
            G.add_edge(j, i, weight=np.log(a[i][j]) + 0.5)


weights = [G[u][v]['weight'] for u, v in G.edges()]

nx.draw(
    G,
    nx.spring_layout(G),
    edges=G.edges(),
    width=2,
    edge_color=weights,
    node_color='#D88D8D',
    edge_cmap=plt.cm.RdPu,
    with_labels=True,
    arrowstyle='->',
    arrowsize=10,
)

plt.show()