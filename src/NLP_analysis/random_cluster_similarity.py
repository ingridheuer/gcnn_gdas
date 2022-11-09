#%%
import numpy as np
import pandas as pd
import random
import nlp_utils
#%%
seed = 16
random.seed(seed)
np.random.seed(seed)

rng = np.random.default_rng()
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

indices = np.arange(num_nodos)
similarity_matrix_0 = nlp_utils.load_lsa_similiarity_matrix_array(lsa_data_path,0)

#%%
def mean_similarity(similarity_matrix, nodos_cluster,N):
    norm = (N**2 - N)/2
    cluster_matrix = similarity_matrix[np.ix_(nodos_cluster,nodos_cluster)]
    meansim = np.sum(cluster_matrix)/norm
    return round(meansim, 2)

def random_cluster_sim(num_nodos,N,similarity_matrix):
    # random_cluster = np.random.choice(indices, N, replace=False)
    # random_cluster = rng.choice(indices,N,replace=False)
    random_cluster = rng.choice(num_nodos,N,replace=False)
    # meansim = nlp_utils.mean_similarity(similarity_matrix,random_cluster)
    meansim = mean_similarity(similarity_matrix,random_cluster,N)
    return meansim

def p_value(iters,size,mean_sim_cluster,similarity_matrix):
    counts = 0
    for _ in range(iters):
        meansim_random = random_cluster_sim(num_nodos,size,similarity_matrix)
        if meansim_random > mean_sim_cluster:
            counts +=1
    pvalue = counts/iters
    return pvalue

#%%

iters = 10
aver = infomap_data.apply(lambda x: p_value(iters,x["tamaño"].astype(int),x["mean_sim_lsa_0"],similarity_matrix_0),axis=1)

#%%

