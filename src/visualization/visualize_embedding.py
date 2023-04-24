#%%
import copy
import torch
import itertools
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA

import sys
sys.path.append("..")
from models.base_model import base_model
#%%
data_folder = "../../data/processed/graph_data_nohubs/"
models_folder = "../../data/models/"
experiments_folder = "../../data/experiments/design_space_experiment/"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#%%
def load_data(folder_path,load_test = False):
    if load_test:
        names = ["train","validation","test"]
    else:
        names = ["train","validation"]
    datasets = []
    for name in names:
        path = folder_path+name+".pt"
        datasets.append(torch.load(path))
    
    return datasets

def initialize_features(data,feature,dim,inplace=False):
    if inplace:
        data_object = data
    else:
        data_object = copy.copy(data)
    for nodetype, store in data_object.node_items():
        if feature == "random":
            data_object[nodetype].x = torch.rand(store["num_nodes"],dim)
        if feature == "ones":
            data_object[nodetype].x = torch.ones(store["num_nodes"],dim)
    return data_object

def load_model(state_dict,params,metadata):
    model = base_model(params,metadata)
    model.load_state_dict(state_dict)
    return model

def load_experiment(eid,date,metadata):
    """date format: d_m_y"""
    df_path = f"{experiments_folder}experiment_{date}.parquet"
    weights_path = f"{experiments_folder}experiment_{eid}_{date}__.pth"

    df = pd.read_parquet(df_path)
    #TODO: this is only temporal, remove after fix
    df["conv_type"] = df.conv_type.apply(lambda x: x.split(".")[-1].rstrip("\'>"))
    df["activation"] = torch.nn.LeakyReLU
    params = df.loc[eid].to_dict()
    weights = torch.load(weights_path,map_location=torch.device(device))

    model = base_model(params,metadata)
    model.load_state_dict(weights)

    return model,params

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

@torch.no_grad()
def get_encodings(model,data):
    
    model.eval()
    encodings = model.encode(data)
    gene_encodings = encodings["gene_protein"]
    disease_encodings = encodings["disease"]

    return gene_encodings, disease_encodings

def plot_pca(gene_encodings,disease_encodings,title,n_components,plot_components):

    z1 = disease_encodings.detach().cpu().numpy()
    z2 = gene_encodings.detach().cpu().numpy()

    z = np.concatenate([z1,z2])

    num_diseases = z1.shape[0]
    num_genes = z2.shape[0]

    pca = PCA(n_components=n_components)
    components = pca.fit_transform(z)
    fig = px.scatter(components, x=plot_components[0], y=plot_components[1], color=['b']*num_diseases + ['r']*num_genes, title=title,hover_name=node_names)

    fig.show()

def plot_pca_3D(gene_encodings,disease_encodings,title,n_components,plot_components):

    z1 = disease_encodings.detach().cpu().numpy()
    z2 = gene_encodings.detach().cpu().numpy()

    z = np.concatenate([z1,z2])

    num_diseases = z1.shape[0]
    num_genes = z2.shape[0]

    pca = PCA(n_components=n_components)
    components = pca.fit_transform(z)
    fig = px.scatter_3d(components, x=plot_components[0], y=plot_components[1],z=plot_components[2], color=['b']*num_diseases + ['r']*num_genes, title=title,hover_name=node_names)

    fig.show()

def plot_pca_with_communities(gene_encodings,disease_encodings,title,n_components,plot_components,node_info,partition="comunidades_louvain"):

    z1 = disease_encodings.detach().cpu().numpy()
    z2 = gene_encodings.detach().cpu().numpy()

    z = np.concatenate([z1,z2])

    pca = PCA(n_components=n_components)
    components = pca.fit_transform(z)

    node_clusters = node_info[["node_index",partition]].dropna()
    node_map_series = pd.Series(node_map["disease"],name="tensor_index")
    node_clusters = node_clusters.merge(node_map_series,left_on="node_index",right_index=True,how="right").sort_values(by="tensor_index").fillna(-1)

    df = pd.DataFrame(components)
    df["node_names"] = node_names
    df = df.merge(node_clusters[["tensor_index",partition]],left_index=True,right_on="tensor_index",how="left").reset_index(drop=True).fillna(-2)
    df[partition] = df[partition].astype(str)
    
    fig = px.scatter(df, x=plot_components[0], y=plot_components[1], color=partition, title=title,hover_name="node_names")
    fig.show()

def plot_pca_all_types(encodings_dict,title,n_components,plot_components):

    encodings = [tensor.detach().cpu().numpy() for tensor in encodings_dict.values()]
    z = np.concatenate(encodings)

    pca = PCA(n_components=n_components)
    components = pca.fit_transform(z)
    component_df = pd.DataFrame(components)

    sub_dfs = []
    for node_type in encodings_dict.keys():
        sub_df = node_data[node_data.node_type == node_type]
        node_map_series = pd.Series(node_map[node_type],name="tensor_index")
        sub_df = sub_df.merge(node_map_series,left_on="node_index",right_index=True,how="right").sort_values(by="tensor_index").reset_index(drop=True)

        sub_dfs.append(sub_df)

    df = pd.concat(sub_dfs,ignore_index=True)
    df = df.merge(component_df,left_index=True,right_index=True)

    fig = px.scatter(df, x=plot_components[0], y=plot_components[1], color="node_type", title=title,hover_name="node_name")

    fig.show()
#%%
results = pd.read_parquet(experiments_folder+"experiment_18_04_23.parquet").sort_values(by="auc",ascending=False)

train_data, val_data = load_data(data_folder+"split_dataset/")
node_data,node_map = load_node_csv(data_folder+"nohub_graph_nodes.csv","node_index","node_type")
node_names = node_data[(node_data.node_type == "disease") | (node_data.node_type == "gene_protein")].sort_values(by="node_type").node_name.values
node_info = pd.read_csv(data_folder+"nohub_graph_node_data.csv")
#%%
eid = 10
date = "18_04_23"
model,params = load_experiment(eid,date,train_data.metadata())
train_data = initialize_features(train_data,params["feature_type"],params["feature_dim"])
val_data = initialize_features(train_data,params["feature_type"],params["feature_dim"])
disease_x = train_data["disease"].x
gene_x = train_data["gene_protein"].x
gene_encodings,disease_encodings = get_encodings(model,train_data)
#%%
plot_pca(gene_x,disease_x,"Vector de features de genes y enfermedades (input al modelo)",2,(0,1))
#%%
plot_pca(gene_encodings,disease_encodings,"Encodings de genes y enfermedades (output del modelo sin decoder)",2,(0,1))
#%%
plot_pca_3D(gene_encodings,disease_encodings,"aver",3,(0,1,2))
#%%
plot_pca_with_communities(gene_encodings,disease_encodings,"aver",2,(0,1),node_info)
#%%
plot_pca_all_types(model.encode(train_data),"aver",2,(0,1))
#%%
