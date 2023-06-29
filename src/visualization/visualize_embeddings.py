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
train_data, val_data = training_utils.load_data(
    data_args["dataset_folder_path"])
node_csv, node_map = training_utils.load_node_csv(
    data_args["node_data_path"], "node_index", "node_type")
node_info = pd.read_csv(data_args["node_info_path"])
node_names = node_csv[(node_csv.node_type == "disease") | (
    node_csv.node_type == "gene_protein")].sort_values(by="node_type").node_name.values

model_args = viz_config["model"]
model = training_utils.load_model(
    model_args["weights_path"], model_args["model_type"], model_args["supervision_types"], train_data.metadata())

train_data = training_utils.initialize_features(
    train_data, model_args["feature_type"], model_args["feature_dim"])
val_data = training_utils.initialize_features(
    train_data, model_args["feature_type"], model_args["feature_dim"])
    
encodings_dict = training_utils.get_encodings(model, train_data)
# %%
# def plot_pca_3D(gene_encodings,disease_encodings,title,n_components,plot_components):

#     z1 = disease_encodings.detach().cpu().numpy()
#     z2 = gene_encodings.detach().cpu().numpy()
#     gene_encodings = encodings["gene_protein"]
#     disease_encodings = encodings["disease"]

#     return gene_encodings, disease_encodings
#     z = np.concatenate([z1,z2])

#     num_diseases = z1.shape[0]
#     num_genes = z2.shape[0]

#     pca = PCA(n_components=n_components)
#     components = pca.fit_transform(z)
#     fig = px.scatter_3d(components, x=plot_components[0], y=plot_components[1],z=plot_components[2], color=['b']*num_diseases + ['r']*num_genes, title=title,hover_name=node_names)

#     fig.show()


# def plot_umap(encodings_dict,n_components,plot_components,title,colors):
#     node_info = pd.read_csv(data_folder+"nohub_graph_node_data.csv")
#     encodings = [tensor.detach().cpu().numpy() for tensor in encodings_dict.values()]
#     z = np.concatenate(encodings)
#     umap = UMAP(n_components=n_components, init='random', random_state=0)
#     proj = umap.fit_transform(z)
#     proj_df = pd.DataFrame(proj)

#     sub_dfs = []
#     for node_type in encodings_dict.keys():
#         sub_df = node_info[node_info.node_type == node_type]
#         node_map_series = pd.Series(node_map[node_type],name="tensor_index")
#         sub_df = sub_df.merge(node_map_series,left_on="node_index",right_index=True,how="right").sort_values(by="tensor_index").reset_index(drop=True)

#         sub_dfs.append(sub_df)

#     df = pd.concat(sub_dfs,ignore_index=True)
#     df = df.merge(proj_df,left_index=True,right_index=True).fillna(-1).astype({"comunidades_infomap":str,"comunidades_louvain":str})

#     fig = px.scatter(df, x=plot_components[0], y=plot_components[1], color=colors, title=title,hover_name="node_name")

#     fig.show()

# def plot_tsne(encodings_dict,n_components,plot_components,title,colors):
#     node_info = pd.read_csv(data_folder+"nohub_graph_node_data.csv")
#     encodings = [tensor.detach().cpu().numpy() for tensor in encodings_dict.values()]
#     z = np.concatenate(encodings)
#     tsne = TSNE(n_components=n_components, random_state=0)
#     proj = tsne.fit_transform(z)
#     proj_df = pd.DataFrame(proj)

#     sub_dfs = []
#     for node_type in encodings_dict.keys():
#         sub_df = node_info[node_info.node_type == node_type]
#         node_map_series = pd.Series(node_map[node_type],name="tensor_index")
#         sub_df = sub_df.merge(node_map_series,left_on="node_index",right_index=True,how="right").sort_values(by="tensor_index").reset_index(drop=True)

#         sub_dfs.append(sub_df)

#     df = pd.concat(sub_dfs,ignore_index=True)
#     df = df.merge(proj_df,left_index=True,right_index=True).fillna(-1).astype({"comunidades_infomap":str,"comunidades_louvain":str})

#     fig = px.scatter(df, x=plot_components[0], y=plot_components[1], color=colors, title=title,hover_name="node_name")

#     fig.show()

# %%
def get_tensor_index_df(node_data, node_map, node_info, encodings_dict):
    sub_dfs = []
    for node_type in encodings_dict.keys():
        sub_df = node_data[node_data.node_type == node_type]
        node_map_series = pd.Series(node_map[node_type], name="tensor_index")
        sub_df = sub_df.merge(node_map_series, left_on="node_index", right_index=True,
                              how="right").sort_values(by="tensor_index").reset_index()

        sub_dfs.append(sub_df)
    tensor_df = pd.concat(sub_dfs, ignore_index=True)
    df = pd.merge(tensor_df, node_info[["node_index", "comunidades_infomap", "comunidades_louvain",
                  "degree_gda", "degree_pp", "degree_dd"]], on="node_index").fillna(-2)
    df = df.astype({"comunidades_louvain": str, "comunidades_infomap": str})
    df["total_degree"] = df.degree_pp + df.degree_gda + df.degree_dd
    return df


def plot_pca(node_data, node_map, node_info, encodings_dict, title, n_components, plot_components, color="node_type"):

    encodings = [tensor.detach().cpu().numpy()
                 for tensor in encodings_dict.values()]
    z = np.concatenate(encodings)

    scaler = StandardScaler().fit(z)
    z = scaler.transform(z)

    pca = PCA(n_components=n_components)
    components = pca.fit_transform(z)
    component_df = pd.DataFrame(components)

    tensor_df = get_tensor_index_df(
        node_data, node_map, node_info, encodings_dict)
    df = pd.merge(component_df, tensor_df, left_index=True, right_index=True)
    df[["degree_gda", "degree_pp", "degree_dd", "total_degree"]] = np.log(
        df[["degree_gda", "degree_pp", "degree_dd", "total_degree"]])

    fig = px.scatter(df, x=plot_components[0], y=plot_components[1],
                     color=color, title=title, hover_name="node_name")

    fig.show()


def plot_pca_3D(node_data, node_map, node_info, encodings_dict, title, n_components, plot_components, color="node_type"):

    encodings = [tensor.detach().cpu().numpy()
                 for tensor in encodings_dict.values()]
    z = np.concatenate(encodings)

    scaler = StandardScaler().fit(z)
    z = scaler.transform(z)

    pca = PCA(n_components=n_components)
    components = pca.fit_transform(z)
    component_df = pd.DataFrame(components)

    tensor_df = get_tensor_index_df(
        node_data, node_map, node_info, encodings_dict)
    df = pd.merge(component_df, tensor_df, left_index=True, right_index=True)
    df[["degree_gda", "degree_pp", "degree_dd", "total_degree"]] = np.log(
        df[["degree_gda", "degree_pp", "degree_dd", "total_degree"]])

    fig = px.scatter_3d(df, x=plot_components[0], y=plot_components[1],
                        z=plot_components[2], color=color, title=title, hover_name="node_name")

    fig.show()


def get_tsne(node_data, node_map, node_info, encodings_dict, n_components):
    encodings = [tensor.detach().cpu().numpy()
                 for tensor in encodings_dict.values()]
    z = np.concatenate(encodings)

    scaler = StandardScaler().fit(z)
    z = scaler.transform(z)

    tsne = TSNE(n_components=n_components,
                random_state=viz_config["misc"]["seed"])
    proj = tsne.fit_transform(z)
    proj_df = pd.DataFrame(proj)

    tensor_df = get_tensor_index_df(
        node_data, node_map, node_info, encodings_dict)
    df = pd.merge(proj_df, tensor_df, left_index=True, right_index=True)
    df[["degree_gda", "degree_pp", "degree_dd", "total_degree"]] = np.log(
        df[["degree_gda", "degree_pp", "degree_dd", "total_degree"]])

    return df


def plot_df(df, title, plot_components, colors):
    fig = px.scatter(df, x=plot_components[0], y=plot_components[1],
                     color=colors, title=title, hover_name="node_name")
    fig.show()


def plot_tsne(node_data, node_map, node_info, encodings_dict, title, n_components, plot_components, colors):

    df = get_tsne(node_data, node_map, node_info, encodings_dict, n_components)
    plot_df(df, title, plot_components, colors)

def get_umap(node_data, node_map, node_info, encodings_dict, n_components):
    encodings = [tensor.detach().cpu().numpy()
                 for tensor in encodings_dict.values()]
    z = np.concatenate(encodings)

    scaler = StandardScaler().fit(z)
    z = scaler.transform(z)

    umap = UMAP(n_components=n_components, init='random', random_state=viz_config["misc"]["seed"])
    proj = umap.fit_transform(z)
    proj_df = pd.DataFrame(proj)

    tensor_df = get_tensor_index_df(
        node_data, node_map, node_info, encodings_dict)
    df = pd.merge(proj_df, tensor_df, left_index=True, right_index=True)
    df[["degree_gda", "degree_pp", "degree_dd", "total_degree"]] = np.log(
        df[["degree_gda", "degree_pp", "degree_dd", "total_degree"]])

    return df
# %%
node_data_args = [node_csv, node_map, node_info, encodings_dict]

plot_pca(*node_data_args, "PCA. Color según grado GDA",
            2, [0, 1], "degree_gda")
# %%
# plot_tsne(*node_data_args, "TSNE", 2, [0, 1], "degree_gda")
# %%
tsne_df = get_tsne(*node_data_args, 2)
umap_df = get_umap(*node_data_args,2)
# %%
plot_df(tsne_df, "aver", [0, 1], "node_type")
#%%
plot_df(umap_df,"umap",[0,1],"comunidades_infomap")
