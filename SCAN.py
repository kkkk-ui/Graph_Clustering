import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import matplotlib.colors as mcolors
import time

node_df = pd.read_csv("Nodes.csv")
edge_df = pd.read_csv("Edges.csv")

G = nx.Graph()

node_list = []
edge_list = []

for node, cluster in zip(node_df["NodeID"], node_df["cluster"]):
    node_list.append(node)
    G.add_node(node)
    G.nodes[node]["cluster"] = cluster

for node_1, node_2 in zip(edge_df["Source"], edge_df["Target"]):
    edge_list.append((node_1, node_2))
    G.add_edge(node_1, node_2)

neighbors = {
    v: set(G.neighbors(v)) | {v}
    for v in G.nodes
}

EPS = 0.8
MU = 3

e_neighbors = {}

# BFS ダメ（最短経路等）
# def neighborhood(node):
#     subgraph_nodes = nx.single_source_shortest_path_length(G, node, cutoff=1).keys()
#     return subgraph_nodes

def sigma(node_1, node_2):
    inter = len(neighbors[node_1] & neighbors[node_2])
    return inter / np.sqrt(len(neighbors[node_1]) * len(neighbors[node_2]))

def e_neighborhood(node_1):
    if node_1 in e_neighbors:
        return e_neighbors[node_1]

    eps_n = [
        node_2 for node_2 in neighbors[node_1]
        if sigma(node_1, node_2) >= EPS
    ]
    e_neighbors[node_1] = eps_n
    return eps_n
        
def core(node):
    if len(e_neighborhood(node)) >= MU:
        return True
    else:
        return False
    
# def dir_reach(node_1, node_2):
#     if core(node_1) and (node_2 in e_neighborhood(node_1)):
#         return True
#     else:
#         return False
    
# def reach(node_1, node_2):
#     try:
#         path = nx.shortest_path(G, node_1, node_2)
#         return True
    
#     except nx.NetworkXNoPath:
#         return False

# def connect(node_1, node_2):
#     difference_set = set(node_list)-set([node_1, node_2])
#     node_3 = np.random.choice(list(difference_set))
#     if reach(node_3, node_1) and reach(node_3, node_2):
#         return True
#     else:
#         return False


# ================================================================================
# SCAN プログラム
start = time.time()
clusterID = 0
for node in node_list:
    if G.nodes[node]["cluster"] != "unclassified":
        continue

    print(f"node:{node}")
    if core(node):
        clusterID += 1
        G.nodes[node]["cluster"] = clusterID
        Q = e_neighborhood(node)

        while len(Q) != 0:
            node_1 = Q[0]
            R = []
            
            # for node_2 in node_list:
            #     if dir_reach(node_1, node_2):
            #         R.append(node_2)

            if core(node_1):
                for node_2 in e_neighborhood(node_1):
                    R.append(node_2)    

            for node_2 in R:
                old_label = G.nodes[node_2]["cluster"]

                if old_label in ("unclassified", "non-member"):
                    G.nodes[node_2]["cluster"] = clusterID

                if old_label == "unclassified":
                    Q.append(node_2)

            Q.remove(node_1)

    else:
        G.nodes[node]["cluster"] = "non-member"

non_member_nodes = [
    node for node, attributes in G.nodes.data() if attributes.get("cluster") == "non-member"
]

for node in non_member_nodes:
    neighbor_nodes = list(G.neighbors(node))

    neighbor_cluster_ids = set()
    for nb in neighbor_nodes:
        label = G.nodes[nb]["cluster"]
        if label not in ("unclassified", "non-member", "hub", "outlier"):
            neighbor_cluster_ids.add(label)

    if len(neighbor_cluster_ids) >= 2:
        G.nodes[node]["cluster"] = "hub"
    else:
        G.nodes[node]["cluster"] = "outlier"
end = time.time()
print(end - start)
# ================================================================================


# ================================================================================
# 可視化
print(len(G.edges))
all_labels = [attr.get("cluster") for _, attr in G.nodes.data() if "cluster" in attr]
unique_labels = list(set(all_labels))

cmap = plt.cm.get_cmap('tab10') 
num_labels = len(unique_labels)

label_to_color = {}
for i, label in enumerate(unique_labels):
    label_to_color[label] = mcolors.rgb2hex(cmap(i / num_labels)[:3]) 

node_colors = []

for node in G.nodes():
    node_label = G.nodes[node]["cluster"]
    color = label_to_color.get(node_label, 'lightgray') 
    node_colors.append(color)

plt.figure(figsize=(10, 10))

pos = nx.spring_layout(G)

nx.draw_networkx_nodes(G, pos, node_size=50, node_color=node_colors, alpha=0.8) 
nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.1)

legend_handles = []
for label in unique_labels:
    color = label_to_color[label]
    display_label = f'Cluster {label}' if isinstance(label, (int, float)) and label > 0 else str(label).capitalize()
    
    handle = plt.Line2D(
        [0], [0], 
        marker='o', 
        color='w', 
        label=display_label,
        markerfacecolor=color, 
        markersize=10
    )
    legend_handles.append(handle)

plt.legend(handles=legend_handles, loc='upper left', title="Node Labels")
plt.axis('off') 
plt.show()
# ================================================================================