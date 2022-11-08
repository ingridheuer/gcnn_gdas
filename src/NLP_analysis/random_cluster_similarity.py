#%%
import numpy as np
import pandas as pd
import random

import config
import nlp_utils
#%%
data_processed = "../../data/processed/"
data_interim = "../../data/interim/"
data_external = "../../data/external/"
graph_data = data_processed + "graph_data_nohubs/"
lsa_data_path = graph_data+"LSA_data/"
lsa_reports = "../../reports/reports_nohubs/analisis_lsa/"


graph_node_data = pd.read_csv(graph_data+"nohub_graph_node_data.csv")

infomap_data = pd.read_csv(lsa_reports+"infomap_meansim.csv")
louvain_data = pd.read_csv(lsa_reports+"louvain_meansim.csv")

infomap_clusters, louvain_clusters = nlp_utils.get_cluster_dataframes(graph_node_data)

num_nodos = len(graph_node_data[graph_node_data.degree_dd !=0]["node_index"].values)
#%%
# similarity_matrix = nlp_utils.load_lsa_similiarity_matrices(lsa_data_path)
similarity_matrix = nlp_utils.load_lsa_similiarity_matrices_sp(lsa_data_path)
#%%
def mean_similarity_np(similarity_matrix, nodos_cluster):
    cluster_matrix = similarity_matrix[np.ix_(nodos_cluster,nodos_cluster)]
    indices = np.triu_indices_from(cluster_matrix,1)
    values = cluster_matrix[indices]
    return round(np.mean(values), 2)

def random_cluster_sim(num_nodos,N,similarity_matrix):
    random_cluster = np.random.choice(range(num_nodos),N)
    # meansim = nlp_utils.mean_similarity(similarity_matrix,random_cluster)
    meansim = mean_similarity_np(similarity_matrix,random_cluster)
    return meansim

def p_value(iters,size,mean_sim_cluster,similarity_matrix):
    counts = 0
    for i_ in range(iters):
        meansim_random = random_cluster_sim(num_nodos,size,similarity_matrix)
        if meansim_random > mean_sim_cluster:
            counts +=1
    pvalue = counts/iters
    return pvalue
#%%
p_values = []
iters = 1000
for row in infomap_data.itertuples(index=False):
    cluster = row[0]
    size = row[1]
    mean_sim = row[2:]
    p_values.append(p_value(iters,size,mean_sim[0],similarity_matrix[0]))
#%%
aver = infomap_data.apply(lambda x: p_value(10,x["tamaño"].astype(int),x["mean_sim_lsa_0"],similarity_matrix[0]),axis=1)
#%%
p_value(1000,50,0.5,aver)
#%%
aver = similarity_matrix[0].to_numpy()
#%%
aver = similarity_matrix[0]