#%%
import numpy as np
import pandas as pd
import random
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
lsa_data_path = graph_data+"LSA_data/"
lsa_reports = "../../reports/reports_nohubs/analisis_lsa/"


graph_node_data = pd.read_csv(graph_data+"nohub_graph_node_data.csv")
infomap_clusters, louvain_clusters = nlp_utils.get_cluster_dataframes(graph_node_data)
infomap_list, louvain_list = nlp_utils.get_cluster_nodelists(graph_node_data)

#%%
similarity_matrix = nlp_utils.load_lsa_similiarity_matrices(lsa_data_path)
#%%
# def mean_similarity(similarity_matrix, nodos_cluster):
#     cluster_matrix = similarity_matrix.loc[nodos_cluster,nodos_cluster].values
#     indices = np.triu_indices_from(cluster_matrix,1)
#     values = cluster_matrix[indices]
#     return round(np.mean(values), 2)

def mean_similarity(similarity_matrix, nodos_cluster,N):
    norm = (N**2 - N)/2
    cluster_matrix = similarity_matrix.loc[nodos_cluster,nodos_cluster].values
    meansim = np.sum(cluster_matrix)/norm
    return round(meansim, 2)

#%%
infomap_meansim = []
louvain_meansim = []

for i in range(4):
    infomap_series = infomap_list.apply(lambda x: mean_similarity(similarity_matrix[i], x,len(x))).rename(f"mean_sim_lsa_{i}")
    louvain_series = louvain_list.apply(lambda x: mean_similarity(similarity_matrix[i], x,len(x))).rename(f"mean_sim_lsa_{i}")

    infomap_meansim.append(infomap_series)
    louvain_meansim.append(louvain_series)

infomap_results = pd.concat(infomap_meansim, axis=1).reset_index()
louvain_results = pd.concat(louvain_meansim, axis=1).reset_index()

infomap_results = pd.merge(infomap_clusters, infomap_results, left_on="comunidad", right_on="comunidades_infomap").drop(columns="comunidades_infomap")
louvain_results = pd.merge(louvain_clusters, louvain_results, left_on="comunidad", right_on="comunidades_louvain").drop(columns="comunidades_louvain")
#%%
infomap_results.to_csv(lsa_reports+"infomap_meansim.csv",index=False)
louvain_results.to_csv(lsa_reports+"louvain_meansim.csv", index=False)