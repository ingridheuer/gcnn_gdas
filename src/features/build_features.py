# %%
import pandas as pd
import numpy as np
import torch
from scipy import sparse
from sklearn.preprocessing import MinMaxScaler, StandardScaler

seed = 5

data_folder = "../../data/processed/graph_data_nohubs/"
split_folder = data_folder + f"/merged_types/split_dataset/seed_{seed}/"
feature_data_folder = "../../data/processed/feature_data/"
tensor_df = pd.read_csv(
    split_folder+"tensor_df.csv", index_col=0
)


def load_sparse_dataframe(matrix_path, row_path):
    mat = sparse.load_npz(matrix_path)
    row = np.loadtxt(row_path)
    df = pd.DataFrame.sparse.from_spmatrix(mat, index=row)
    return df


def fill_missing_with_mean(feature_matrix, node_type, node_data=tensor_df):
    node_count = tensor_df.node_type.value_counts()[node_type]

    global_mean = feature_matrix.mean(axis=0).values
    has_feature = feature_matrix.index.values.astype(int)
    all_index = node_data[node_data.node_type == node_type].node_index.values
    no_feature = list(set(all_index) - set(has_feature))

    global_mean_matrix = np.tile(global_mean, (len(no_feature), 1))
    global_mean_df = pd.DataFrame(
        global_mean_matrix, index=no_feature, columns=feature_matrix.columns)
    full_feature_df = (
        pd.concat([feature_matrix, global_mean_df])
        .reset_index()
        .rename(columns={"index": "node_index"})
    )

    full_feature_df = (
        pd.merge(
            full_feature_df,
            node_data[["node_index", "tensor_index"]],
            left_on="node_index",
            right_on="node_index",
        )
        .sort_values(by="tensor_index")
        .drop(columns=["node_index"])
        .set_index("tensor_index")
    )
    assert len(
        full_feature_df) == node_count, f"Lenght of feature tensor {len(full_feature_df)} does not match number of nodes {node_count}"
    return full_feature_df


lsa_matrix_path = data_folder + "/LSA_data/lsa_matrix_0.npz"
index_path = data_folder + "/LSA_data/matrix_index_0.txt"

lsa_matrix = load_sparse_dataframe(lsa_matrix_path, index_path)
lsa_matrix = lsa_matrix.sparse.to_dense()

full_feature_df = fill_missing_with_mean(lsa_matrix, "disease")
lsa_feature_tensor = torch.tensor(full_feature_df.values)
torch.save(lsa_feature_tensor, feature_data_folder+"lsa_features.pt")

# %%
# Normalize and rescale features
scaler = StandardScaler()
scaler.fit(lsa_matrix.values)
scaled_matrix = scaler.transform(lsa_matrix.values)

scaled_lsa_matrix = pd.DataFrame(scaled_matrix, index=lsa_matrix.index)
scaled_feature_df = fill_missing_with_mean(scaled_lsa_matrix, "disease")
scaled_feature_tensor = torch.tensor(scaled_feature_df.values)
torch.save(scaled_feature_tensor, feature_data_folder+"lsa_scaled_features.pt")

minmax_scaler = MinMaxScaler([-1, 1])
minmax_scaler.fit(lsa_matrix.values)
minmax_matrix = minmax_scaler.transform(lsa_matrix.values)

norm_feature_matrix = pd.DataFrame(minmax_matrix, index=lsa_matrix.index)
norm_feature_df = fill_missing_with_mean(norm_feature_matrix, "disease")
norm_feature_tensor = torch.tensor(norm_feature_df.values)
torch.save(norm_feature_tensor, feature_data_folder+"lsa_norm_features.pt")
# %%
# # Build gene features:
# gtex = pd.read_csv(feature_data_folder+"gtex.embedding.d41.tsv", sep="\t")
# scaler = MinMaxScaler([-1,1])
# scaler.fit(gtex.values)
# scaled_values = scaler.transform(gtex.values)
# scaled_df = pd.DataFrame(scaled_values, index=gtex.index, columns=gtex.columns)

# scaled_df_full = fill_missing_with_mean(scaled_df,"gene_protein")
# scaled_full_tensor = torch.tensor(scaled_df_full.values)
# torch.save(scaled_full_tensor,feature_data_folder+"gtex_norm_features.pt")
# %%
gtex_41 = pd.read_csv(feature_data_folder+"gtex.embedding.d41.tsv", sep="\t")
gtex_41 = gtex_41[~gtex_41.index.duplicated(
    keep='first')].reset_index(names="node_id")
genes = tensor_df[tensor_df.node_id.apply(lambda x: x.isnumeric())]
genes_features = pd.merge(gtex_41, genes[["node_id", "node_index"]].astype(
    int), left_on="node_id", right_on="node_id").set_index("node_index").drop(columns="node_id")

scaler = MinMaxScaler([-1, 1])
scaler.fit(genes_features.values)
norm_values = scaler.transform(genes_features.values)
norm_df = pd.DataFrame(
    norm_values, index=genes_features.index, columns=genes_features.columns)

gtex_norm_df = fill_missing_with_mean(norm_df, "gene_protein", tensor_df)
gtex_norm_tensor = torch.tensor(gtex_norm_df.values)
torch.save(gtex_norm_tensor, feature_data_folder+"gtex_norm_features.pt")
# %%
scaler = StandardScaler()
scaler.fit(genes_features.values)
scaled_values = scaler.transform(genes_features.values)
scaled_df = pd.DataFrame(
    scaled_values, index=genes_features.index, columns=genes_features.columns)

gtex_scaled_df = fill_missing_with_mean(scaled_df, "gene_protein", tensor_df)
gtex_scaled_tensor = torch.tensor(gtex_scaled_df.values)
torch.save(gtex_norm_tensor, feature_data_folder+"gtex_scaled_features.pt")
