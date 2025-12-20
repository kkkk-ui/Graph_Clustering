import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import matplotlib.colors as mcolors
import GraphKernelFunc as gkf
from grakel import Graph
import time
import cProfile
import pstats
# ================================================================================
# 関数
def nx_to_grakel_graph(neighbors):
    sublabel = {n: str(G.nodes[n]["label"]) for n in neighbors}
    edge_list = []
    for n in neighbors:
        for v in G.neighbors(n):
            if v in neighbors and n < v: # 重複（u,v と v,u）を避けるため 
                edge_list.append((n, v))

    return Graph(edge_list, node_labels=sublabel)

def core_score(v):
    neigh = set(G.neighbors(v)) | {v}
    # neigh_subgraph = nx_to_grakel_graph(neigh)

    classified = [u for u in neigh if G.nodes[u]["cluster"] != "unclassified"]

    if len(classified) == 0:
        return 0.0
    
    # gkf.GraphkernelFunc.k_func_wl(nx_to_grakel_graph(set(G.neighbors(u)) | {u}), neigh_subgraph, 1) 
    same = sum(1 for u in classified 
               if G.nodes[u]["cluster"] == G.nodes[v]["cluster"])
    return same / len(classified)

def update_representative_cache(changed_nodes):
    """
    changed_nodes: グラフ構造が変わったノード集合
    """
    for r in list(representative_graphs.keys()):
        # 代表 r の局所構造が影響を受けるか？
        if r in changed_nodes or any(n in changed_nodes for n in G.neighbors(r)):
            subgraph_nodes = set(G.neighbors(r)) | {r}
            representative_graphs[r] = nx_to_grakel_graph(subgraph_nodes)
# ================================================================================

# ================================================================================
# 初期設定（グラフ生成等）
node_df = pd.read_csv("Nodes.csv")
edge_df = pd.read_csv("Edges.csv")

G = nx.Graph()

node_list = []
edge_list = []

for node in zip(node_df["NodeID"], node_df["label"], node_df["cluster"]):
    node_list.append(node[0])
    G.add_node(node[0])
    G.nodes[node[0]]["label"] = node[1]
    G.nodes[node[0]]["cluster"] = node[2]

for edge in zip(edge_df["Source"], edge_df["Target"]):
    edge_list.append(edge)
    G.add_edge(edge[0],edge[1])
# ================================================================================

# ================================================================================
# Adaptive clustering プログラム
# プロファイリングの実行
profiler = cProfile.Profile()
profiler.enable()
# start = time.time()
nodeID_dict = []
representative_graphs = {}
clusterID = 0
sigma = 0.8
while True:
    print(nodeID_dict)
    unclassified_nodes = [n for n, data in G.nodes(data=True) if data.get("cluster") == "unclassified"]

    if unclassified_nodes:
        picked = np.random.choice(unclassified_nodes)
        node = picked
        # print(f"node:{picked}")
    else:
        break

    if clusterID == 0:
        clusterID += 1

        nodeID_dict.append(node)
        subgraph_nodes = set(G.neighbors(node)) | {node}
        representative_graphs[node] = nx_to_grakel_graph(subgraph_nodes)

        G.nodes[node]["cluster"] = clusterID
    else:
        subgraph_nodes = set(G.neighbors(node)) | {node}
        if len(subgraph_nodes) == 1:
            G.nodes[node]["cluster"] = "non-member"
            continue
        subgraph = nx_to_grakel_graph(subgraph_nodes)

        coh_max = 0
        u = None
        best_idx = None
        for idx, j in enumerate(nodeID_dict):
            # representative_nodes = set(G.neighbors(j)) | {j}
            # if len(representative_nodes) == 1:
            #     continue
            # representative = nx_to_grakel_graph(representative_nodes)

            coh = gkf.GraphkernelFunc.k_func_wl(subgraph, representative_graphs[j],  1)
            # coh = gkf.GraphkernelFunc.k_func_vh(subgraph, representative)
            if(coh > coh_max):
                coh_max = coh
                u = j
                best_idx = idx

        if(coh_max < sigma):
            clusterID += 1
            nodeID_dict.append(node)

            representative_graphs[node] = nx_to_grakel_graph(subgraph_nodes)

            G.nodes[node]["cluster"] = clusterID

            for neighborhood in subgraph_nodes:
                if G.nodes[neighborhood]["cluster"] != "unclassified":
                    continue
                
                neighbor_subgraph_nodes = set(G.neighbors(neighborhood)) | {neighborhood}
                if len(neighbor_subgraph_nodes) == 1:
                    continue
                neighbor_subgraph = nx_to_grakel_graph(neighbor_subgraph_nodes)

                coh = gkf.GraphkernelFunc.k_func_wl(neighbor_subgraph, subgraph, 1)
                # coh = gkf.GraphkernelFunc.k_func_vh(neighbor_subgraph, subgraph)
                    
                if(coh >= sigma):
                    G.nodes[neighborhood]["cluster"] = G.nodes[node]["cluster"]

        elif(coh_max >= sigma):
            G.nodes[node]["cluster"] = G.nodes[u]["cluster"]

            for neighborhood in subgraph_nodes:
                if G.nodes[neighborhood]["cluster"] != "unclassified":
                    continue
                
                neighbor_subgraph_nodes = set(G.neighbors(neighborhood)) | {neighborhood}
                if len(neighbor_subgraph_nodes) == 1:
                    continue
                neighbor_subgraph = nx_to_grakel_graph(neighbor_subgraph_nodes)

                coh = gkf.GraphkernelFunc.k_func_wl(neighbor_subgraph, subgraph, 1)
                # coh = gkf.GraphkernelFunc.k_func_vh(neighbor_subgraph, subgraph)
                    
                if(coh >= sigma):
                    G.nodes[neighborhood]["cluster"] = G.nodes[node]["cluster"]
            
            if (core_score(node) > core_score(u)) :
                nodeID_dict[best_idx] = node
                representative_graphs.pop(u)
                representative_graphs[node] = nx_to_grakel_graph(subgraph_nodes)
        
# end = time.time()
# print(end - start)
profiler.disable()
stats = pstats.Stats(profiler).sort_stats('cumtime') # 累積時間でソート
stats.print_stats(20) # 上位20項目を表示
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

node_sizes = []
for node in G.nodes():
    if node in nodeID_dict:
        node_sizes.append(300)   # 代表ノード
    else:
        node_sizes.append(50)   # 通常ノード

plt.figure(figsize=(10, 10))

pos = nx.spring_layout(G)

nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8) 
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