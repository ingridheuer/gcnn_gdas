# %%
import pandas as pd
import numpy as np
import torch
from scipy import sparse

seed = 5

data_folder = "../../data/processed/graph_data_nohubs/"
split_folder = data_folder + f"/merged_types/split_dataset/seed_{seed}/"
tensor_df = pd.read_csv(
    split_folder+"tensor_df.csv", index_col=0
)


def load_sparse_dataframe(matrix_path, row_path):
    mat = sparse.load_npz(matrix_path)
    row = np.loadtxt(row_path)
    df = pd.DataFrame.sparse.from_spmatrix(mat, index=row)
    return df


lsa_matrix_path = data_folder + "/LSA_data/lsa_matrix_0.npz"
index_path = data_folder + "/LSA_data/matrix_index_0.txt"

lsa_matrix = load_sparse_dataframe(lsa_matrix_path, index_path)
lsa_matrix = lsa_matrix.sparse.to_dense()

global_mean = lsa_matrix.mean(axis=0).values
lsa_matrix.index = lsa_matrix.index.astype(int)
has_feature = lsa_matrix.index.values
no_feature = list(
    set(tensor_df[tensor_df.node_type == "disease"].node_index.values)
    - set(has_feature)
)

global_mean_matrix = np.tile(global_mean, (len(no_feature), 1))
global_mean_df = pd.DataFrame(global_mean_matrix, index=no_feature)
full_feature_df = (
    pd.concat([lsa_matrix, global_mean_df])
    .reset_index()
    .rename(columns={"index": "node_index"})
)

disease_only = tensor_df[tensor_df.node_type == "disease"]
full_feature_df = (
    pd.merge(
        full_feature_df,
        tensor_df[["node_index", "tensor_index"]],
        left_on="node_index",
        right_on="node_index",
    )
    .sort_values(by="tensor_index")
    .drop(columns=["node_index"])
    .set_index("tensor_index")
)

lsa_feature_tensor = torch.tensor(full_feature_df.values)
torch.save(lsa_feature_tensor, split_folder+"lsa_features.pt")

# %%
