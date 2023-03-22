#%%
import pandas as pd 
from torch_geometric.data import HeteroData
from torch_geometric.transforms import RandomLinkSplit
import torch
import networkx as nx
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
#%%
# Load data from csv and create heterodata object
data_folder = "../../data/processed/graph_data_nohubs/"
node_data, node_map = load_node_csv(data_folder+"nohub_graph_nodes.csv","node_index","node_type")
edge_data, edge_index = load_edge_csv(data_folder+"nohub_graph_edge_data.csv","x_index","y_index",node_map,"edge_type","x_type","y_type")
data = create_heterodata(node_map,edge_index)
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#%%
# Demonstrate link split on karateclub graph
def attributes_from_pd(G:nx.Graph,df:pd.DataFrame,attributes:dict,indexcol=None):
    """Dados un grafo G y un dataframe df con atributos de sus nodos, especificamos los atributos
    que queremos agregar a los nodos en un diccionario con formato {nombre_columna:nombre_atributo}. 
    La función arma un diccionario con los atributos y el nombre que le queremos poner, indexado con el identificador de nodo que elegimos 
    y los asigna a los nodos del grafo"""
    for attribute,name in attributes.items():
        if indexcol == None:
            nx.set_node_attributes(G,pd.Series(df[attribute]).to_dict(),name)
        else:
            nx.set_node_attributes(G,pd.Series(df.set_index(indexcol)[attribute]).to_dict(),name)

G = nx.from_pandas_edgelist(edge_data,source="x_index",target="y_index",edge_attr="edge_type")
G_attributes = {"node_type":"node_type","node_name":"node_name","node_id":"node_id"}
attributes_from_pd(G,node_data,G_attributes)
#%%
node = 600
g = nx.ego_graph(G,node,2)
g.number_of_nodes()
#%%
def draw_graph(g):
    node_color_rule = {"disease":"black","bert_group":"black","gene_protein":"grey","complex":"grey","pathway":"grey"}
    edge_color_rule = {"gda":"green","disease_disease":"red","ppi":"blue","pathway_protein":"blue","forms_complex":"blue"}

    node_colors = [node_color_rule[node_type] for node_type in nx.get_node_attributes(g,"node_type").values()]
    edge_colors = [edge_color_rule[edge_type] for edge_type in nx.get_edge_attributes(g,"edge_type").values()]

    nx.draw(g,pos=nx.kamada_kawai_layout(g),node_color=node_colors,edge_color=edge_colors,node_size=100)
#%%
nodelist = {}
data.subgraph()