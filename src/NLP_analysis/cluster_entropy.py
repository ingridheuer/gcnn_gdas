#%%
import numpy as np
import pandas as pd
import random
from scipy.stats import entropy
import nlp_utils
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
entropy_reports = "../../reports/reports_nohubs/analisis_tfidf/entropy/"

graph_node_data = pd.read_csv(graph_data+"nohub_graph_node_data.csv")

infomap_clusters, louvain_clusters = nlp_utils.get_cluster_dataframes(graph_node_data)
#%%
path_infomap = graph_data + "tfidf_infomap/"
path_louvain = graph_data + "tfidf_louvain/"

tfidf_infomap, tfidf_louvain = nlp_utils.load_cluster_matrices(path_infomap,path_louvain,4)

tfidf_infomap_dense = [mat.sparse.to_dense() for mat in tfidf_infomap]
tfidf_louvain_dense = [mat.sparse.to_dense() for mat in tfidf_louvain]
#%%
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

def max_mean_ratio(arr, use_nonzero=True):
    if use_nonzero:
        arr = arr[np.nonzero(arr)]
    max = np.max(arr)
    mean = np.mean(arr)
    ratio = round(max/mean)
    return ratio

def alpha_ij(arr):
    nonzero = arr[np.nonzero(arr)]
    k = len(nonzero)

    alpha_values = np.round((1 - nonzero)**(k-1),2)
    indices = np.argsort(-alpha_values)
    alpha_values = alpha_values[indices]

    return alpha_values,indices
#%%
entropias_infomap = []
for i, mat in enumerate(tfidf_infomap_dense):
    name = "entropia_"+str(i)
    entropia_series = mat.apply(lambda x: get_entropy(x), axis=1, raw=True).rename(name)
    entropias_infomap.append(entropia_series)

entropias_louvain = []
for i, mat in enumerate(tfidf_louvain_dense):
    name = "entropia_"+str(i)
    entropia_series = mat.apply(lambda x: get_entropy(x), axis=1, raw=True).rename(name)
    entropias_louvain.append(entropia_series)

entropias_infomap_df = pd.DataFrame(entropias_infomap).T
entropias_louvain_df = pd.DataFrame(entropias_louvain).T

results_infomap = pd.merge(infomap_clusters, entropias_infomap_df, left_on="comunidad",right_index=True)
results_louvain = pd.merge(louvain_clusters, entropias_louvain_df, left_on="comunidad",right_index=True)

results_infomap.to_csv(entropy_reports+"entropy_infomap.csv",index=False)
results_louvain.to_csv(entropy_reports+"entropy_louvain.csv",index=False)
#%%
