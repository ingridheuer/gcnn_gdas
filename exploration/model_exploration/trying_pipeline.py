#%%
import pandas as pd 
from torch_geometric.transforms import RandomLinkSplit
import torch
import networkx as nx
import pipeline_utils
#%%
# Load data from csv and create heterodata object
data_folder = "../../data/processed/graph_data_nohubs/"
node_data, node_map = pipeline_utils.load_node_csv(data_folder+"nohub_graph_nodes.csv","node_index","node_type")
edge_data, edge_index = pipeline_utils.load_edge_csv(data_folder+"nohub_graph_edge_data.csv","x_index","y_index",node_map,"edge_type","x_type","y_type")
data = pipeline_utils.create_heterodata(node_map,edge_index)
#%%
#Initialize features

def initialize_features(data_object,feature,dim):
    for nodetype, store in data_object.node_items():
        if feature == "random":
            data_object[nodetype].x = torch.rand(store["num_nodes"],dim)
        if feature == "ones":
            data_object[nodetype].x = torch.ones(store["num_nodes"],dim)
    return data_object


#%%
#Split the dataset
edge_types, rev_edge_types = pipeline_utils.get_reverse_types(data.edge_types)
split_transform = RandomLinkSplit(num_val=0.3, num_test=0.3, is_undirected=True, add_negative_train_samples=True, disjoint_train_ratio=0.2,edge_types=edge_types,rev_edge_types=rev_edge_types)
train_data, val_data, test_data = split_transform(data)
#%%