#%%
import pandas as pd
import numpy as np
from scipy import sparse

import networkx as nx
import random
import network_utils as util
import itertools as it
#%%
seed = 16
random.seed(seed)
np.random.seed(seed)
#%%
data_processed = "../../data/processed/"
data_interim = "../../data/interim/"
data_external = "../../data/external/"
graph_data = data_processed + "graph_data_nohubs/"

reports_genes = "../../reports/reports_nohubs/analisis_red_genes/"

graph_node_data = pd.read_csv(graph_data+"nohub_graph_node_data.csv")
nodos_gda = graph_node_data[(graph_node_data.degree_gda !=0) & (graph_node_data.degree_dd !=0)]

infomap_nodelist,louvain_nodelist = util.get_cluster_nodelists(graph_node_data)

infomap_results = pd.read_csv(reports_genes+"infomap_gene_sim.csv")
louvain_results = pd.read_csv(reports_genes+"louvain_gene_sim.csv")

overlap_sim = util.load_sparse_dataframe(reports_genes+"overlap_3.npz",reports_genes+"index_matrices_jaccard.txt",reports_genes+"index_matrices_jaccard.txt",False)
#%%
def keep_gda_only(cluster_nodelist,nodos_gda):
    in_gda = set(cluster_nodelist)&nodos_gda
    return list(in_gda)

def group_by_degree(data_df,num_groups):
    df = data_df.copy()
    spacing = 2 ** np.arange(num_groups)
    bins = np.digitize(df.degree_gda.values, spacing)
    df["bins"] = bins
    df = df.sort_values(by="bins")

    return df

def mean_similarity(similarity_matrix, nodos_cluster,N):
    norm = (N**2 - N)/2
    cluster_matrix = similarity_matrix.loc[nodos_cluster,nodos_cluster].values
    meansim = np.sum(cluster_matrix)/norm
    return round(meansim, 2)

def sample_from_bin(num,bin,df=nodos_gda):
    sample = df[df.bins == bin].sample(num)
    return sample.node_index.values

def sample_from_dist(dist_index,dist_values):
    sample = []
    for bin,num in zip(dist_index,dist_values):
        sample.append(sample_from_bin(num,bin))

    unnested_sample = list(it.chain.from_iterable(sample))
    return unnested_sample

def random_cluster_sim(dist,similarity_matrix):
    # val = df.loc[df.comunidades_infomap == cluster, metric].values[0]
    # nodos = nodos_gda.set_index("node_index").loc[cluster_nodelist[cluster]]
    # dist = nodos.bins.value_counts()
    random_sample = sample_from_dist(dist.index, dist.values)
    random_val = mean_similarity(similarity_matrix,random_sample,len(random_sample))

    return random_val

def p_value(nodelist,iters,mean_sim_cluster,similarity_matrix):
    if len(nodelist) < 2:
        print("Cluster with less than 2 nodes in gda - skip")
        pvalue = 1
    else:
        counts = 0
        nodos = nodos_gda.set_index("node_index").loc[nodelist]
        dist = nodos.bins.value_counts()
        for _ in range(iters):
            meansim_random = random_cluster_sim(dist,similarity_matrix)
            if meansim_random > mean_sim_cluster:
                counts +=1
        pvalue = counts/iters
    return pvalue
#%%
num_groups = 11
nodos_gda = group_by_degree(nodos_gda,num_groups) 
infomap_clusters = infomap_nodelist.apply(lambda x: keep_gda_only(x,set(nodos_gda.node_index.values)))
louvain_clusters = louvain_nodelist.apply(lambda x: keep_gda_only(x,set(nodos_gda.node_index.values)))

data_infomap = pd.concat([infomap_results[["comunidades_infomap","mean_sim_overlap_3"]],infomap_clusters],axis=1)
data_louvain = pd.concat([louvain_results[["comunidades_louvain","mean_sim_overlap_3"]],louvain_clusters],axis=1)
#%%
iters = 1000
print("Infomap pvalues...")
infomap_pvalues = data_infomap.apply(lambda x: p_value(x["node_index"],iters,x["mean_sim_overlap_3"],overlap_sim), axis=1).rename("pvalue_overlap_3")

print("Louvain pvalues ...")
louvain_pvalues = data_louvain.apply(lambda x: p_value(x["node_index"],iters,x["mean_sim_overlap_3"],overlap_sim), axis=1).rename("pvalue_overlap_3")

pvalue_results_infomap = pd.concat([infomap_results[["comunidades_infomap","mean_sim_overlap_3"]],infomap_pvalues],axis=1)
pvalue_results_louvain = pd.concat([louvain_results[["comunidades_louvain","mean_sim_overlap_3"]],louvain_pvalues],axis=1)

pvalue_results_infomap.to_csv(reports_genes+"infomap_genesim_pvalues.csv")
pvalue_results_louvain.to_csv(reports_genes+"louvain_genesim_pvalues.csv")

print("Done")
#%%