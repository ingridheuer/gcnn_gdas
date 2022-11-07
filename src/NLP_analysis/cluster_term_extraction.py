#%%
import numpy as np
import pandas as pd
from scipy import sparse
import random
#%%
seed = 16
random.seed(seed)
np.random.seed(seed)
#%%
data_processed = "../../data/processed/"
data_interim = "../../data/interim/"
data_external = "../../data/external/"
graph_data = data_processed + "graph_data_nohubs/"
path_infomap = graph_data + "tfidf_infomap/"
path_louvain = graph_data + "tfidf_louvain/"

graph_node_data = pd.read_csv(graph_data+"nohub_graph_node_data.csv")
tamaños_louvain = graph_node_data.comunidades_louvain.dropna().value_counts()
tamaños_infomap = graph_node_data.comunidades_infomap.dropna().value_counts()

infomap_clusters = pd.DataFrame(tamaños_infomap).reset_index().rename(columns={"index":"comunidad","comunidades_infomap":"tamaño"}).astype({"comunidad":"int"})
louvain_clusters = pd.DataFrame(tamaños_louvain).reset_index().rename(columns={"index":"comunidad","comunidades_louvain":"tamaño"}).astype({"comunidad":"int"})

infomap_ids = infomap_clusters.comunidad.values
louvain_ids = louvain_clusters.comunidad.values

#%%

def load_sparse_dataframe(matrix_path,row_path,column_path):
    mat = sparse.load_npz(matrix_path)
    row = np.loadtxt(row_path)
    col = np.loadtxt(column_path, dtype="str")
    df = pd.DataFrame.sparse.from_spmatrix(mat, index=row, columns=col)
    return df

def load_cluster_matrices(path_infomap:str,path_louvain:str,number_of_matrices):
    tfidf_infomap = []
    tfidf_louvain = []

    for i in range(number_of_matrices):
        mat_path = f"{path_infomap}matriz_tfidf_infomap_{i}.npz"
        row_path = f"{path_infomap}rows_tfidf_infomap_{i}.txt"
        col_path = f"{path_infomap}cols_tfidf_infomap_{i}.txt"
        
        tfidf_infomap.append(load_sparse_dataframe(mat_path,row_path,col_path))

    for i in range(number_of_matrices):
        mat_path = f"{path_louvain}matriz_tfidf_louvain_{i}.npz"
        row_path = f"{path_louvain}rows_tfidf_louvain_{i}.txt"
        col_path = f"{path_louvain}cols_tfidf_louvain_{i}.txt"
        
        tfidf_louvain.append(load_sparse_dataframe(mat_path,row_path,col_path))
    
    return tfidf_infomap, tfidf_louvain

def get_top_terms(matrix,cluster,number_of_terms):
    top_terms = matrix.loc[cluster].sort_values(ascending=False)[0:number_of_terms].index.values.astype(str)
    top_terms_scores = np.around(matrix.loc[cluster].sort_values(ascending=False)[0:number_of_terms].values, decimals=2)
    return top_terms, top_terms_scores

def top_ngrams(matrix_list,matrix_names_list,clusters,number_of_terms):
    all_stats = {}
    for cluster in clusters:
        stats = {}
        for matrix,name in zip(matrix_list,matrix_names_list):
            top_terms,top_scores = get_top_terms(matrix,cluster,number_of_terms)

            stats[f"top_{name}"] = top_terms[0]
            stats[f"top_{name}_score"] = top_scores[0]

            stats[f"top_{number_of_terms}_{name}"] = top_terms
            stats[f"top_{number_of_terms}_{name}_score"] = top_scores

        all_stats[cluster] = stats
    return all_stats

#%%
print("Loading cluster matrices ...")
tfidf_infomap, tfidf_louvain = load_cluster_matrices(path_infomap,path_louvain,4)

tfidf_infomap_dense = [mat.sparse.to_dense() for mat in tfidf_infomap]
tfidf_louvain_dense = [mat.sparse.to_dense() for mat in tfidf_louvain]
#%%
number_of_terms = 5
matrix_list_names = ["monogram","bigram","trigram","mixed"]

print("Finding top terms ...")
infomap_top_terms = top_ngrams(tfidf_infomap_dense,matrix_list_names,infomap_ids,number_of_terms)
louvain_top_terms = top_ngrams(tfidf_louvain_dense,matrix_list_names,louvain_ids,number_of_terms)

infomap_top_terms_df = pd.DataFrame(infomap_top_terms).T.reset_index().rename(columns={"index":"comunidad"})
louvain_top_terms_df = pd.DataFrame(louvain_top_terms).T.reset_index().rename(columns={"index":"comunidad"})
#%%
infomap_top_terms_df.to_pickle(graph_data+"infomap_top_terms.pkl")
louvain_top_terms_df.to_pickle(graph_data+"louvain_top_terms.pkl")

print(f"Top terms done. Data saved to {graph_data}")