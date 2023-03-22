#%%
import pandas as pd 
from torch_geometric.data import HeteroData
from torch_geometric.transforms import RandomLinkSplit
import torch
import networkx as nx
import matplotlib.pyplot as plt
#%%
def load_node_csv(path, index_col,type_col, **kwargs):
    """Returns node dataframe and a dict of mappings for each node type. 
    Each mapping maps from original df index to "heterodata index" { node_type : { dataframe_index : heterodata_index}}"""
    df = pd.read_csv(path, **kwargs,index_col=index_col)
    node_types = df[type_col].unique()
    mappings_dict = dict()
    for node_type in node_types:
        mapping = {index: i for i, index in enumerate(df[df[type_col] == node_type].index.unique())}
        mappings_dict[node_type] = mapping

    return df,mappings_dict

def load_edge_csv(path, src_index_col, dst_index_col, mappings, edge_type_col,src_type_col,dst_type_col, **kwargs):
    """Returns edge dataframe and a dict of edge indexes. Nodes are indexed according to the "heterodata index", using the node mappings from load_node_csv. Edge indexes are tensors of shape [2, num_edges]. Dict is indexed by triplets of shape (src_type, edge_type, dst_type)."""
    df = pd.read_csv(path, **kwargs)
    df["edge_triple"] = list(zip(df[src_type_col],df[edge_type_col], df[dst_type_col]))
    edge_triplets = df["edge_triple"].unique()

    edge_index_dict = dict()
    for edge_triplet in edge_triplets:

        sub_df = df[df.edge_triple == edge_triplet]
        src_type,edge_type,dst_type = edge_triplet

        src_mapping = mappings[src_type]
        dst_mapping = mappings[dst_type]

        src = [src_mapping[index] for index in sub_df[src_index_col]]
        dst = [dst_mapping[index] for index in sub_df[dst_index_col]]
        edge_index = torch.tensor([src, dst])
        edge_index_dict[edge_triplet] = edge_index

    return df, edge_index_dict

def create_heterodata(node_map, edge_index):
    """Initializes HeteroData object from torch_geometric and creates corresponding nodes and edges, without any features"""
    data = HeteroData()
    for node_type,vals in node_map.items():
        # Initialize all node types without features
        data[node_type].num_nodes = len(vals)
    
    for edge_triplet, index in edge_index.items():
        src_type, edge_type, dst_type = edge_triplet
        data[src_type, edge_type, dst_type].edge_index = index
    
    return data

def tensor_to_edgelist(tensor: torch.tensor):
    "Toma un edge_index de shape (2,num_edges) y devuelve una lista de tuplas"
    sources = tensor[0,:].tolist()
    targets = tensor[1,:].tolist()
    edgelist = list(zip(sources,targets))
    return edgelist

def reverse_map(node_map,edge_list,edge_type):
    src_map = {v:k for k,v in node_map[edge_type[0]].items()}
    dst_map = {v:k for k,v in node_map[edge_type[2]].items()}
    mapped_edge_list = [(src_map[n1],dst_map[n2]) for (n1,n2) in edge_list]

    return mapped_edge_list

def inverse_map_heterodata(data,node_map):
    edge_dict = {}
    for edge_type in data.edge_types:
        type_dict = {}
        if len(data[edge_type].keys()) > 1:
            edge_tensor = data[edge_type]["edge_index"]
            edge_list = tensor_to_edgelist(edge_tensor)
            mapped_edge_list = reverse_map(node_map,edge_list,edge_type)

            labeled_edges_tensor = data[edge_type]["edge_label_index"]
            labeled_edges_list = tensor_to_edgelist(labeled_edges_tensor)
            mapped_labeled_edges_list = reverse_map(node_map,labeled_edges_list,edge_type)

            edge_labels = data[edge_type]["edge_label"].tolist()

            type_dict["message_passing_edges"] = mapped_edge_list
            type_dict["supervision_edges"] = mapped_labeled_edges_list
            type_dict["supervision_labels"] = edge_labels
        else:
            edge_tensor = data[edge_type]["edge_index"]
            edge_list = tensor_to_edgelist(edge_tensor)
            mapped_edge_list = reverse_map(node_map,edge_list,edge_type)

            type_dict["message_passing_edges"] = mapped_edge_list
         
        edge_dict[edge_type] = type_dict
    
    return edge_dict
#%%
data_folder = "../../data/processed/graph_data_nohubs/"
node_data, node_map = load_node_csv(data_folder+"subgraph_nodes.csv","node_index","node_type")
edge_data, edge_index = load_edge_csv(data_folder+"subgraph_edges.csv","x_index","y_index",node_map,"edge_type","x_type","y_type")
data = create_heterodata(node_map,edge_index)
#%%
edge_types = [('disease', 'disease_disease', 'bert_group'),('gene_protein', 'gda', 'disease'),('gene_protein', 'ppi', 'gene_protein')]
rev_edge_types = [('bert_group', 'disease_disease', 'disease'),('disease', 'gda', 'gene_protein'),('gene_protein', 'ppi', 'gene_protein')]

good_transform = RandomLinkSplit(num_val=0.3, num_test=0.3, is_undirected=True, add_negative_train_samples=True, disjoint_train_ratio=0.2,edge_types=edge_types,rev_edge_types=rev_edge_types)

bad_transform = RandomLinkSplit(num_val=0.3, num_test=0.3, is_undirected=True, add_negative_train_samples=True, disjoint_train_ratio=0.2,edge_types=data.edge_types)

g_train, g_val, g_test = good_transform(data)
b_train, b_val, b_test = bad_transform(data)
#%%
# G = nx.DiGraph()
G = nx.MultiDiGraph()
G.add_nodes_from(node_data.index.values)
nx.set_node_attributes(G,node_data["node_type"].to_dict(),"node_type")
#%%
# def add_edges_from_pyg(G,data,node_map):
#     H = G.copy()
#     edge_dict = inverse_map_heterodata(data,node_map)
#     for edge_type,dictionary in edge_dict.items():
#         for subtype, edgelist in dictionary.items():
#             if subtype != "supervision_labels":
#                 if "supervision_labels" in dictionary.keys():
#                     sup_labels = {e:label for e,label in zip(dictionary["supervision_edges"],dictionary["supervision_labels"])}
#                     edges = [(n1,n2,{"subtype":subtype, "type":edge_type[1], "supervision_label":sup_labels[(n1,n2)]}) for (n1,n2) in edgelist]
#                     H.add_edges_from(edges)
#                 else:
#                     edges = [(n1,n2,{"subtype":subtype, "type":edge_type[1], "supervision_label":1}) for  (n1,n2) in edgelist]
#                     H.add_edges_from(edges)
#     return H
#%%
def add_edges_from_pyg(G,data,node_map):
    H = G.copy()
    edge_dict = inverse_map_heterodata(data,node_map)
    for edge_type,dictionary in edge_dict.items():
        message_edges = [(n1,n2,{"subtype":"message_passing_edges", "type":edge_type[1], "supervision_label":1.0}) for (n1,n2) in dictionary["message_passing_edges"]]
        H.add_edges_from(message_edges)

        if "supervision_edges" in dictionary.keys():
            sup_labels = {e:label for e,label in zip(dictionary["supervision_edges"],dictionary["supervision_labels"])}
            supervision_edges = [(n1,n2,{"subtype":"supervision_edges", "type":edge_type[1], "supervision_label":sup_labels[(n1,n2)]}) for (n1,n2) in dictionary["supervision_edges"]]
            H.add_edges_from(supervision_edges)
    return H
#%%
def generate_style_list(G,key,rule_dict,nodes):
    if nodes:
        style_list = [rule_dict[case] for case in nx.get_node_attributes(G,key).values()]
    else:
        style_list = [rule_dict[case] for case in nx.get_edge_attributes(G,key).values()]
    return style_list

def draw_graph(g,ax,node_color_rule,edge_color_rule,edge_line_rule,layout):
    node_colors = generate_style_list(g,node_color_rule[0],node_color_rule[1],True)
    edge_colors = generate_style_list(g,edge_color_rule[0],edge_color_rule[1],False)
    edge_lines = generate_style_list(g,edge_line_rule[0],edge_line_rule[1],False)

    nx.draw(g,ax=ax,pos=layout,node_color=node_colors,edge_color=edge_colors,style=edge_lines,node_size=100,connectionstyle="arc3,rad=0.1")

#%%
def draw_splits(G,split_list,node_map,node_color_rule,edge_color_rule,edge_line_rule):
    fig,ax = plt.subplots(3,1,figsize=(10,30),squeeze=True)
    for i,split in enumerate(split_list):
        g = add_edges_from_pyg(G,split,node_map)
        if i == 0:
            layout = nx.kamada_kawai_layout(g)
        draw_graph(g,ax[i],node_color_rule,edge_color_rule,edge_line_rule,layout)
#%%
good_list = [g_train,g_val,g_test]
bad_list = [b_train,b_val,b_test]
#%%
node_color_rule = {"disease":"black","bert_group":"black","gene_protein":"grey","complex":"grey","pathway":"grey"}
edge_color_rule = {"message_passing_edges":"red","supervision_edges":"blue"}
edge_line_rule = {1.0:"solid",0.0:"dashed"}

draw_splits(G,bad_list,node_map,("node_type",node_color_rule),("subtype",edge_color_rule),("supervision_label",edge_line_rule))
#%%
node_color_rule = {"disease":"black","bert_group":"black","gene_protein":"grey","complex":"grey","pathway":"grey"}
edge_color_rule = {"gda":"green","disease_disease":"red","ppi":"blue","pathway_protein":"blue","forms_complex":"blue"}
edge_line_rule = {"message_passing_edges":"solid","supervision_edges":"dashed"}

draw_splits(G,good_list,node_map,("node_type",node_color_rule),("type",edge_color_rule),("subtype",edge_line_rule))
