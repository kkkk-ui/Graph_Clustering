import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

# グラフのパラメータ設定
N = 300 # ノード数
k = 5   # クラスター数

# クラスターごとのノード数 (合計200になるように分割)
sizes = [60, 70, 60, 50, 60]

# クラスター間の接続確率行列 P
# 対角要素 (クラスター内): 0.8 (密)
# 非対角要素 (クラスター間): 0.01 (疎)
p_in = 0.8
p_out = 0.001
probs = [[p_in, p_out, p_out, p_out, p_out],
         [p_out, p_in, p_out, p_out, p_out],
         [p_out, p_out, p_in, p_out, p_out],
         [p_out, p_out, p_out, p_in, p_out],
         [p_out, p_out, p_out, p_out, p_in]]

# グラフの生成
G = nx.stochastic_block_model(sizes, probs, seed=42)
# seed=42 は再現性のため設定

# ノードの属性設定（色付けのため）
# ノードがどのクラスターに属するかを示すリストを作成
cluster_map = []
for i, size in enumerate(sizes):
    cluster_map.extend([i] * size)

# 描画用のカラーマップ
color_map = {0: 'red', 1: 'red', 2: 'red', 3: 'red', 4: 'red'}
node_colors = [color_map[cluster_map[node]] for node in G.nodes()]

# グラフの可視化
plt.figure(figsize=(10, 10))

# kamada_kawai_layout: クラスター構造が比較的きれいに分離して表示されやすいレイアウト
pos = nx.kamada_kawai_layout(G)

# グラフの描画
nx.draw_networkx_nodes(G, pos, node_size=50, node_color=node_colors, alpha=0.8)
nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.1)

# タイトルと凡例の設定
plt.title(f"Stochastic Block Model (N={N}, K={k})", fontsize=16)

# 凡例用のダミーノードを作成
legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=f'Cluster {i+1} ({sizes[i]} nodes)',
                            markerfacecolor=color, markersize=10)
                  for i, color in color_map.items()]
plt.legend(handles=legend_handles, loc='upper left')

plt.axis('off') # 軸を非表示
plt.show()

print(f"生成されたグラフのノード数: {G.number_of_nodes()}")
print(f"生成されたグラフの辺の数: {G.number_of_edges()}")

# ノードリストとエッジリストの生成と出力
# ノードリストの作成
node_data = []
for node in G.nodes():
    cluster_id = cluster_map[node] + 1   
    node_data.append({
        'NodeID': node,
        'label': cluster_id,             
        'cluster': "unclassified"
    })


node_list_df = pd.DataFrame(node_data)

# エッジリストの作成
# 辺の始点 (Source) と終点 (Target) のノードIDを含める
edge_list_data = []
for u, v in G.edges():
    edge_list_data.append({'Source': u, 'Target': v})

edge_list_df = pd.DataFrame(edge_list_data)

# CSV形式で出力

print("\n" + "="*50)
print("1. ノードリスト (Nodes.csv)")
print("="*50)
node_list_df.to_csv("Nodes.csv", index=False)
print("出力完了")

print("\n" + "="*50)
print("2. エッジリスト (Edges.csv)")
print("="*50)
edge_list_df.to_csv("Edges.csv", index=False)
print("出力完了")