#%%
import networkx as nx
import pandas as pd
import torch
from deepsnap.hetero_graph import HeteroGraph
#%%
data_path = "../../data/processed/"
node_data = pd.read_csv(data_path+"graph_node_table.csv", index_col=0)
edge_data = pd.read_csv(data_path+"graph_edge_table.csv",index_col=0).rename(columns={"relation":"edge_type"})
#%%
D = nx.from_pandas_edgelist(edge_data,source="a_idx",target="b_idx", edge_attr="edge_type", create_using=nx.DiGraph)
nx.set_node_attributes(D,pd.Series(node_data.node_type, index=node_data.node_idx).to_dict(),"node_type")
nx.set_node_attributes(D,pd.Series(node_data.node_name, index=node_data.node_idx).to_dict(),"node_name")

#G = nx.to_undirected(D)
G = D

node_id = 5
print(f"Node {node_id} has properties:", G.nodes(data=True)[node_id])
# %%
edges = list(G.edges())
edge_idx = 123456
n1 = edges[edge_idx][0]
n2 = edges[edge_idx][1]
edge = list(G.edges(data=True))[edge_idx]
print(f"Edge ({edge[0]}, {edge[1]}) has properties:", edge[2])
print(f"Node {n1} has properties:", G.nodes(data=True)[n1])
print(f"Node {n2} has properties:", G.nodes(data=True)[n2])
#%%
G_deepsnap = HeteroGraph(G)
G_deepsnap.edge_types
G_deepsnap.message_types
G_deepsnap.node_types
G_deepsnap.num_nodes()
G_deepsnap.num_edges()
list(G.edges(data=True))[0]
#%%
nx.write_gml(G,data_path+"graph_no_features.gml")