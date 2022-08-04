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
D = nx.from_pandas_edgelist(edge_data,source="a_idx",target="b_idx", edge_attr=["edge_type","YearInitial","YearFinal","score","edge_idx"])
nx.set_node_attributes(D,pd.Series(node_data.node_type, index=node_data.node_idx).to_dict(),"node_type")
nx.set_node_attributes(D,pd.Series(node_data.node_name, index=node_data.node_idx).to_dict(),"node_name")
nx.set_node_attributes(D,pd.Series(node_data.node_idx, index=node_data.node_idx).to_dict(),"node_dataset_idx")
nx.set_node_attributes(D,pd.Series(node_data.disgenet_type, index=node_data.node_idx).to_dict(),"disgenet_type")
nx.set_node_attributes(D,pd.Series(node_data.diseaseClassMSH, index=node_data.node_idx).to_dict(),"diseaseClassMSH")
nx.set_node_attributes(D,pd.Series(node_data.diseaseClassNameMSH, index=node_data.node_idx).to_dict(),"diseaseClassNameMSH")
G = D.to_directed()
#%%
nx.write_gml(G,data_path+"graph_no_features.gml")