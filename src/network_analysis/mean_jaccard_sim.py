#%%
import pandas as pd
import numpy as np
from scipy import sparse

import networkx as nx
import network_utils as util
#%%
data_processed = "../../data/processed/"
data_interim = "../../data/interim/"
data_external = "../../data/external/"
graph_data = data_processed + "graph_data_nohubs/"

reports_genes = "../../reports/reports_nohubs/analisis_red_genes/"

graph_node_data = pd.read_csv(graph_data+"nohub_graph_node_data.csv")
nodos_gda = set(graph_node_data[(graph_node_data.degree_gda != 0) & (graph_node_data.degree_dd != 0)].sort_values(by="node_index").node_index.values)

infomap_clusters, louvain_clusters = util.get_cluster_nodelists(graph_node_data)
infomap_df, louvain_df = util.get_cluster_dataframes(graph_node_data)
#%%
def mean_similarity(similarity_matrix, nodos_cluster,N):
    norm = (N**2 - N)/2
    cluster_matrix = similarity_matrix.loc[nodos_cluster,nodos_cluster].values
    meansim = np.sum(cluster_matrix)/norm
    return round(meansim, 2)

def keep_gda_only(cluster_nodelist,nodos_gda):
    in_gda = set(cluster_nodelist)&nodos_gda
    return list(in_gda)

def get_mean_sim(metric,infomap_clusters,louvain_clusters):
    row_path = reports_genes+"index_matrices_jaccard.txt"
    mat_path = reports_genes+f"{metric}.npz"
    matrix = util.load_sparse_dataframe(mat_path,row_path,row_path,False)
    infomap_meansim = infomap_clusters.apply(lambda x: mean_similarity(matrix, x, len(x))).rename(f"mean_sim_{metric}")
    louvain_meansim = louvain_clusters.apply(lambda x: mean_similarity(matrix, x, len(x))).rename(f"mean_sim_{metric}")
    return infomap_meansim, louvain_meansim
#%%
infomap_clusters = infomap_clusters.apply(lambda x: keep_gda_only(x,nodos_gda))
louvain_clusters = louvain_clusters.apply(lambda x: keep_gda_only(x,nodos_gda))

metrics = ["jaccard_1","overlap_1","jaccard_3","overlap_3"]

infomap_meansim = []
louvain_meansim = []

for metric in metrics:
    infomap_series, louvain_series = get_mean_sim(metric,infomap_clusters,louvain_clusters)
    infomap_meansim.append(infomap_series)
    louvain_meansim.append(louvain_series)
#%%
infomap_results = pd.concat(infomap_meansim, axis=1).reset_index().astype({"comunidades_infomap":int})
louvain_results = pd.concat(louvain_meansim, axis=1).reset_index().astype({"comunidades_louvain":int})
#%%
infomap_results.to_csv(reports_genes+"infomap_gene_sim.csv",index=False)
louvain_results.to_csv(reports_genes+"louvain_gene_sim.csv",index=False)