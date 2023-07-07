# %%
import sys
sys.path.append("..")

from config import viz_config
from models import training_utils
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from umap import UMAP
# %%
# Load data
data_args = viz_config["data"]
datasets, _ = training_utils.load_data(data_args["dataset_folder_path"])
train_data, val_data = datasets

model_args = viz_config["model"]
model = training_utils.load_model(model_args["weights_path"], model_args["model_type"], model_args["supervision_types"], train_data.metadata())

train_data = training_utils.initialize_features(train_data, model_args["feature_type"], model_args["feature_dim"])
val_data = training_utils.initialize_features(train_data, model_args["feature_type"], model_args["feature_dim"])
    
encodings_dict = training_utils.get_encodings(model, train_data)
tensor_df = pd.read_csv(data_args["tensor_df_path"],index_col=0).fillna(-2).astype({"comunidades_infomap":str, "comunidades_louvain":str})
# %%
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
                     color=color, title=title, hover_name="node_name")

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
                random_state=viz_config["misc"]["seed"])
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

    umap = UMAP(n_components=n_components, init='random', random_state=viz_config["misc"]["seed"])
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
node_data_args = [tensor_df, encodings_dict]

plot_pca(*node_data_args, "PCA. Color según grado GDA",
            2, [0, 1], "degree_gda")
#%%
plot_pca_3D(tensor_df,encodings_dict,"aver",3,[0,1,2],"degree_gda")
# %%
# plot_tsne(*node_data_args, "TSNE", 2, [0, 1], "degree_gda")
# %%
tsne_df = get_tsne(*node_data_args, 2)
umap_df = get_umap(*node_data_args,2)
# %%
plot_df(tsne_df, "aver", [0, 1], "node_type")
#%%
plot_df(umap_df,"umap",[0,1],"comunidades_infomap")
#%%
pca_df = get_pca_df(tensor_df,encodings_dict,2)
#%%
communities = [0,-2]
plot_df(pca_df[(pca_df.comunidades_infomap == "300.0")|(pca_df.comunidades_louvain == "-2.0")],"aver",[0,1],"comunidades_infomap")
# %%
