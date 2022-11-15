#%%
import pandas as pd
import networkx as nx
import itertools as it
from scipy import sparse
import numpy as np
#%%
def get_node_neighbor_sets(node_list:list,G):
    node_sets = {node:set(G.neighbors(node)) for node in node_list}
    return node_sets

def get_node_neighbor_lists(node_list:list,G):
    node_lists = {n:list(G.neighbors(n)) for n in node_list}
    return node_lists

def attributes_from_pd(G:nx.Graph,df:pd.DataFrame,attributes:dict,indexcol):
    """Dados un grafo G y un dataframe df con atributos de sus nodos, especificamos los atributos
    que queremos agregar a los nodos en un diccionario con formato {nombre_columna:nombre_atributo}. 
    La función arma un diccionario con los atributos y el nombre que le queremos poner, indexado con el identificador de nodo que elegimos 
    y los asigna a los nodos del grafo"""
    for attribute,name in attributes.items():
        nx.set_node_attributes(G,pd.Series(df.set_index(indexcol)[attribute]).to_dict(),name)
    
def neighbors_from_list(node_list,G):
    neighbor_lists = [G.neighbors(n) for n in node_list] #list of lists
    # unnested_set = {item for sublist in neighbor_lists for item in sublist} #set of nodes in lists
    unnested_set = set(it.chain.from_iterable(neighbor_lists))
    return unnested_set

def get_cluster_nodelists(graph_node_data):
    infomap_list = graph_node_data[["node_index","comunidades_infomap"]].dropna().groupby("comunidades_infomap")["node_index"].apply(list)
    louvain_list = graph_node_data[["node_index","comunidades_louvain"]].dropna().groupby("comunidades_louvain")["node_index"].apply(list)

    return infomap_list, louvain_list

def get_cluster_dataframes(graph_node_data):
    tamaños_louvain = graph_node_data.comunidades_louvain.dropna().value_counts()
    tamaños_infomap = graph_node_data.comunidades_infomap.dropna().value_counts()

    infomap_clusters = pd.DataFrame(tamaños_infomap).reset_index().rename(columns={"index":"comunidad","comunidades_infomap":"tamaño"}).astype({"comunidad":"int"})
    louvain_clusters = pd.DataFrame(tamaños_louvain).reset_index().rename(columns={"index":"comunidad","comunidades_louvain":"tamaño"}).astype({"comunidad":"int"})

    return infomap_clusters, louvain_clusters

def load_sparse_dataframe(matrix_path,row_path,column_path,str_cols=True):
    mat = sparse.load_npz(matrix_path)
    row = np.loadtxt(row_path)
    if str_cols:
        col = np.loadtxt(column_path, dtype="str")
    else:
        col = np.loadtxt(column_path)

    df = pd.DataFrame.sparse.from_spmatrix(mat, index=row, columns=col)
    return df