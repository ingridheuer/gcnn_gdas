# %%
# from config import viz_config
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from umap import UMAP
import torch
from torch_geometric import seed_everything

import sys
sys.path.append("..")
from models import training_utils,  base_model, final_model

seed = 4
seed_everything(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# %%
# Load data
# data_args = viz_config["data"]
# model_args = viz_config["model"]
# experiment_results = pd.read_parquet(data_args["results_folder_path"]+"experiment_18_04_23.parquet")
# datasets, _ = training_utils.load_data(data_args["dataset_folder_path"])
# train_data, val_data = datasets
#%%
# model_params = experiment_results.drop(columns=["auc","delta","activation","curve_data"]).loc[34].to_dict()
# model_params = model_params|{"conv_type":"SAGEConv"}

# model = base_model.base_model(model_params,train_data.metadata(),[("gene_protein","gda","disease")])
# weights = torch.load(model_args["weights_path"],map_location=device)
# model.load_state_dict(weights)
# model = training_utils.load_model(model_args["weights_path"], model_args["model_type"], model_args["supervision_types"], train_data.metadata())

# if model_args["feature_type"] != "lsa":
#     train_data = training_utils.initialize_features(train_data, model_args["feature_type"], model_args["feature_dim"])
#     val_data = training_utils.initialize_features(train_data, model_args["feature_type"], model_args["feature_dim"])
# else:
#     train_data = training_utils.initialize_features(train_data, model_args["feature_type"], model_args["feature_dim"],data_args["dataset_folder_path"])
#     val_data = training_utils.initialize_features(train_data, model_args["feature_type"], model_args["feature_dim"],data_args["dataset_folder_path"])
#%%
data_folder = "../../data/processed/graph_data_nohubs/merged_types/split_dataset/"
models_folder = "../../models/final_model/"
feature_folder = "../../data/processed/feature_data/"
#%%
#load data
dataset, node_map = training_utils.load_data(data_folder+f"seed_{seed}/",load_test=True)
test_data = dataset[2]
node_df = pd.read_csv(data_folder+f"seed_{seed}/tensor_df.csv",index_col=0).set_index("node_index",drop=True)

with open(f"{models_folder}training_parameters.pickle", 'rb') as handle:
    params = pickle.load(handle)

#initialize features in test data
test_data  = training_utils.initialize_features(test_data,params["feature_type"],params["feature_dim"],feature_folder)

#load model
weights_path = models_folder+f"seeds/final_model_{seed}.pth"
weights = torch.load(weights_path)
model = final_model.Model(test_data.metadata(),[("gene_protein","gda","disease")])
model.load_state_dict(weights)
encodings_dict = training_utils.get_encodings(model,test_data)

node_df = node_df.fillna(-2).astype({"comunidades_infomap":str, "comunidades_louvain":str})
#%%
def plot_pca(tensor_df, encodings_dict, title, n_components, plot_components, color="node_type"):

    encodings = [tensor.detach().cpu().numpy() for tensor in encodings_dict.values()]
    z = np.concatenate(encodings)

    scaler = StandardScaler().fit(z)
    z = scaler.transform(z)

    pca = PCA(n_components=n_components)
    components = pca.fit_transform(z)
    component_df = pd.DataFrame(components)

    df = pd.merge(component_df, tensor_df, left_index=True, right_index=True)
    df[["degree_gda", "degree_pp", "degree_dd", "total_degree"]] = np.log(
        df[["degree_gda", "degree_pp", "degree_dd", "total_degree"]])

    fig = px.scatter(df, x=plot_components[0], y=plot_components[1],
                     color=color, title=title, hover_name="node_name",width=800,height=500, labels = {"degree_gda":"Log(k)"})

    fig.show()

def get_pca_df(tensor_df,encodings_dict,n_components):
    encodings = [tensor.detach().cpu().numpy() for tensor in encodings_dict.values()]
    z = np.concatenate(encodings)

    scaler = StandardScaler().fit(z)
    z = scaler.transform(z)

    pca = PCA(n_components=n_components)
    components = pca.fit_transform(z)
    component_df = pd.DataFrame(components)

    df = pd.merge(component_df, tensor_df, left_index=True, right_index=True)
    df[["degree_gda", "degree_pp", "degree_dd", "total_degree"]] = np.log(
        df[["degree_gda", "degree_pp", "degree_dd", "total_degree"]])
    
    return df

def plot_pca_3D(tensor_df, encodings_dict, title, n_components, plot_components, color="node_type"):

    encodings = [tensor.detach().cpu().numpy()
                 for tensor in encodings_dict.values()]
    z = np.concatenate(encodings)

    scaler = StandardScaler().fit(z)
    z = scaler.transform(z)

    pca = PCA(n_components=n_components)
    components = pca.fit_transform(z)
    component_df = pd.DataFrame(components)

    df = pd.merge(component_df, tensor_df, left_index=True, right_index=True)
    df[["degree_gda", "degree_pp", "degree_dd", "total_degree"]] = np.log(
        df[["degree_gda", "degree_pp", "degree_dd", "total_degree"]])

    fig = px.scatter_3d(df, x=plot_components[0], y=plot_components[1],
                        z=plot_components[2], color=color, title=title, hover_name="node_name")

    fig.show()


def get_tsne(tensor_df, encodings_dict, n_components):
    encodings = [tensor.detach().cpu().numpy()
                 for tensor in encodings_dict.values()]
    z = np.concatenate(encodings)

    scaler = StandardScaler().fit(z)
    z = scaler.transform(z)

    tsne = TSNE(n_components=n_components,
                random_state=seed)
    proj = tsne.fit_transform(z)
    proj_df = pd.DataFrame(proj)

    df = pd.merge(proj_df, tensor_df, left_index=True, right_index=True)
    df[["degree_gda", "degree_pp", "degree_dd", "total_degree"]] = np.log(
        df[["degree_gda", "degree_pp", "degree_dd", "total_degree"]])

    return df

def get_umap(tensor_df, encodings_dict, n_components):
    encodings = [tensor.detach().cpu().numpy()
                 for tensor in encodings_dict.values()]
    z = np.concatenate(encodings)

    scaler = StandardScaler().fit(z)
    z = scaler.transform(z)

    umap = UMAP(n_components=n_components, init='random', random_state=seed)
    proj = umap.fit_transform(z)
    proj_df = pd.DataFrame(proj)

    df = pd.merge(proj_df, tensor_df, left_index=True, right_index=True)
    df[["degree_gda", "degree_pp", "degree_dd", "total_degree"]] = np.log(
        df[["degree_gda", "degree_pp", "degree_dd", "total_degree"]])

    return df

def plot_df(df, title, plot_components, colors):
    fig = px.scatter(df, x=plot_components[0], y=plot_components[1],
                     color=colors, title=title, hover_name="node_name")
    fig.show()
# %%
plot_pca(node_df,encodings_dict, "PCA. Color según tipo de nodo",2, [0, 1], "degree_gda")
#%%
plot_pca_3D(node_df,encodings_dict,"aver",3,[0,1,2],"node_type")
# %%
# plot_tsne(*node_data_args, "TSNE", 2, [0, 1], "degree_gda")
# %%
tsne_df = get_tsne(node_df.reset_index(),encodings_dict, 2)
umap_df = get_umap(node_df.reset_index(),encodings_dict,2)
pca_df = get_pca_df(node_df.reset_index(),encodings_dict,2)
# %%
plot_df(tsne_df, "TSNE. Color según grado tipo de nodo", [0, 1], "node_type")
#%%
plot_df(umap_df,"UMAP. Color según grado GDA",[0,1],"node_type")
#%%
comu = 70
plot_df(pca_df[(pca_df.comunidades_infomap == str(float(comu)))|(pca_df.comunidades_louvain == "-2.0")],"aver",[0,1],"comunidades_infomap")
# %%
comu = 150
plot_df(umap_df[(umap_df.comunidades_infomap == str(float(comu)))|(umap_df.comunidades_infomap == "-2.0")],"aver",[0,1],"comunidades_infomap")

# %%
comu = 700
plot_df(tsne_df[(tsne_df.comunidades_infomap == str(float(comu)))|(tsne_df.comunidades_infomap == "-2.0")],"aver",[0,1],"comunidades_infomap")
#%%
plot_df(pca_df,"avor",[0,1],"degree_gda")
# %%
plot_pca_3D(node_df.reset_index(),encodings_dict,"aver",3,[0,1,2],"degree_gda")