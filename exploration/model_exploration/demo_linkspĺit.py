#%%
from torch_geometric.transforms import RandomLinkSplit
import torch
import networkx as nx
import matplotlib.pyplot as plt
import pipeline_utils
#%%
def tensor_to_edgelist(tensor: torch.tensor):
    sources = tensor[0,:].tolist()
    targets = tensor[1,:].tolist()
    edgelist = list(zip(sources,targets))
    return edgelist

def reverse_map(node_map,edge_list,edge_type):
    """Maps edge dictionary from pyg Heterodata back into the original node indexes from the dataframe"""
    src_map = {v:k for k,v in node_map[edge_type[0]].items()}
    dst_map = {v:k for k,v in node_map[edge_type[2]].items()}
    mapped_edge_list = [(src_map[n1],dst_map[n2]) for (n1,n2) in edge_list]

    return mapped_edge_list

def inverse_map_heterodata(data,node_map):
    """Maps full edge data from pyg Heterodata back into the original node indexes from the dataframe"""
    edge_dict = {}
    for edge_type in data.edge_types:
        type_dict = {}
        edge_tensor = data[edge_type]["edge_index"]
        edge_list = tensor_to_edgelist(edge_tensor)
        mapped_edge_list = reverse_map(node_map,edge_list,edge_type)

        type_dict["message_passing_edges"] = mapped_edge_list

        if "edge_label_index" in data[edge_type].keys():
            labeled_edges_tensor = data[edge_type]["edge_label_index"]
            labeled_edges_list = tensor_to_edgelist(labeled_edges_tensor)
            mapped_labeled_edges_list = reverse_map(node_map,labeled_edges_list,edge_type)

            edge_labels = data[edge_type]["edge_label"].tolist()

            type_dict["supervision_edges"] = mapped_labeled_edges_list
            type_dict["supervision_labels"] = edge_labels
 
        edge_dict[edge_type] = type_dict
    
    return edge_dict

def add_edges_from_pyg(G,data,node_map):
    """Imports edges from pyg Heterodata splits into a Networkx graph (assumes both graphs have the same nodes)"""
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

def draw_splits(G,split_list,node_map,node_color_rule,edge_color_rule,edge_line_rule):
    fig,ax = plt.subplots(3,1,figsize=(10,30),squeeze=True, dpi=300)
    for i,split in enumerate(split_list):
        g = add_edges_from_pyg(G,split,node_map)
        if i == 0:
            layout = nx.kamada_kawai_layout(g)
        draw_graph(g,ax[i],node_color_rule,edge_color_rule,edge_line_rule,layout)
#%%
#Load edge and node csv files
data_folder = "../../data/processed/graph_data_nohubs/"

node_data, node_map = pipeline_utils.load_node_csv(data_folder+"subgraph_nodes.csv","node_index","node_type")
edge_data, edge_index = pipeline_utils.load_edge_csv(data_folder+"subgraph_edges.csv","x_index","y_index",node_map,"edge_type","x_type","y_type")

#Initialize heterodata object
data = pipeline_utils.create_heterodata(node_map,edge_index)
#%%
#Generate edge type lists to ensure even splitting between both edge directions
edge_types, rev_edge_types = pipeline_utils.get_reverse_types(data.edge_types)

#Split the dataset
#Example of a good split for link prediction: disjoint split, consider reverse edges (prevents reversed edge from leaking into the split)
good_transform = RandomLinkSplit(num_val=0.3, num_test=0.3, is_undirected=True, add_negative_train_samples=True, disjoint_train_ratio=0.2,edge_types=edge_types,rev_edge_types=rev_edge_types)

#Exapmple of a bad split for link prediction: message passing and supervision edges can be shared (non disjoint), dont consider reverse edges
bad_transform = RandomLinkSplit(num_val=0.3, num_test=0.3, is_undirected=True, add_negative_train_samples=True,edge_types=data.edge_types)

g_train, g_val, g_test = good_transform(data)
b_train, b_val, b_test = bad_transform(data)

good_list = [g_train,g_val,g_test]
bad_list = [b_train,b_val,b_test]
#%%
#Initialize graph without edges
G = nx.MultiDiGraph()
G.add_nodes_from(node_data.index.values)
nx.set_node_attributes(G,node_data["node_type"].to_dict(),"node_type")
#%%
#Plot splits with edges colored by subtype (message passing or supervision) and line style by supervision label
node_color_rule = {"disease":"black","bert_group":"brown","gene_protein":"grey","complex":"grey","pathway":"grey"}
edge_color_rule = {"message_passing_edges":"red","supervision_edges":"blue"}
edge_line_rule = {1.0:"solid",0.0:"dashed"}

draw_splits(G,good_list,node_map,("node_type",node_color_rule),("subtype",edge_color_rule),("supervision_label",edge_line_rule))
#%%
#Plot splits with edges colored by type (gda,ppi,disease_disease, etc) and line style by supervision label
node_color_rule = {"disease":"black","bert_group":"black","gene_protein":"grey","complex":"grey","pathway":"grey"}
edge_color_rule = {"gda":"green","ppi":"blue","disease_disease":"red"}
edge_line_rule = {1.0:"solid",0.0:"dashed"}

draw_splits(G,good_list,node_map,("node_type",node_color_rule),("type",edge_color_rule),("supervision_label",edge_line_rule))
#%%
