#%%
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import entropy
#%%
def get_cluster_dataframes(graph_node_data):
    tamaños_louvain = graph_node_data.comunidades_louvain.dropna().value_counts()
    tamaños_infomap = graph_node_data.comunidades_infomap.dropna().value_counts()

    infomap_clusters = pd.DataFrame(tamaños_infomap).reset_index().rename(columns={"index":"comunidad","comunidades_infomap":"tamaño"}).astype({"comunidad":"int"})
    louvain_clusters = pd.DataFrame(tamaños_louvain).reset_index().rename(columns={"index":"comunidad","comunidades_louvain":"tamaño"}).astype({"comunidad":"int"})

    return infomap_clusters, louvain_clusters

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

def load_node_matrices(path:str):
    document_term_matrix = []
    
    for i in range(4):
        mat_path = f"{path}matriz_nodos_tfidf_{i}.npz"
        row_path = f"{path}rows_tfidf_nodos_{i}.txt"
        col_path = f"{path}cols_tfidf_nodos_{i}.txt"

        document_term_matrix.append(load_sparse_dataframe(mat_path, row_path, col_path))
    
    return document_term_matrix

def get_entropy(arr, use_nonzero=False,max_norm=True):
    if use_nonzero:
        values = arr[np.nonzero(arr)]
    else:
        values = arr
    
    if max_norm:
        max_entropy = np.log2(len(values))
        S = round(entropy(values, base=2)/max_entropy , 2)
    else:
        S = round(entropy(values, base=2) , 2)

    return S