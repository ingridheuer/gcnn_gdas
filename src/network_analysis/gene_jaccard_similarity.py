#%%
import pandas as pd
import numpy as np
from scipy import sparse

import networkx as nx
import random
import network_utils as util
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
# graph_edge_data = pd.read_csv(graph_data+"nohub_graph_edge_data.csv")

G = nx.read_gml(graph_data+"nohub_gda_network.gml", destringizer=int)
#%%
# nodos_gda = pd.DataFrame(dict(G.nodes(data=True))).T.reset_index().rename(columns={"index":"node_index"})
# nodos_enfermedad = nodos_gda.loc[nodos_gda.node_type == "disease", "node_index"].sort_values().values
nodos_enfermedad = graph_node_data[(graph_node_data.degree_gda != 0) & (graph_node_data.degree_dd != 0)].sort_values(by="node_index").node_index.values

np.savetxt(reports_genes+"index_matrices_jaccard.txt",nodos_enfermedad)
#%%
def get_2_order_sets(node_list,G):
    first_order = util.get_node_neighbor_sets(node_list,G)
    second_order = {n:util.neighbors_from_list(first_order[n],G) for n in node_list}
    set_union = {n:first.union(second_order[n]) for n,first in first_order.items()}
    
    return first_order, second_order, set_union

def jaccard(set1,set2):
    intersection = len(set1&set2)
    union = len(set1|set2)
    return round(intersection/union,2)

def overlap(set1,set2):
    intersection = len(set1&set2)
    min_set = min([len(set1),len(set2)])
    return round(intersection/min_set,2)

def get_coef_matrix(metric,nodos_enfermedad,conjuntos_enfermedad,return_sparse=False):
    matrix = np.zeros((len(nodos_enfermedad), len(nodos_enfermedad)))
    upper_tri = np.triu_indices_from(matrix,k=1)

    print("Computing metric between pairs ...")
    for i,j in zip(*upper_tri):
        set_i = conjuntos_enfermedad[nodos_enfermedad[i]]
        set_j = conjuntos_enfermedad[nodos_enfermedad[j]]
        coef = metric(set_i,set_j)
        matrix[i][j] = coef
    
    if return_sparse:
        print("Dense to sparse ...")
        matrix = sparse.csr_matrix(matrix)

    print("Done")
    return matrix

#%%
first_order,second_order, both_orders = get_2_order_sets(nodos_enfermedad,G)
jaccard_matrix_1 = get_coef_matrix(jaccard,nodos_enfermedad,first_order,True)
overlap_matrix_1 = get_coef_matrix(overlap,nodos_enfermedad,first_order,True)

sparse.save_npz(reports_genes+"jaccard_1.npz",jaccard_matrix_1)
sparse.save_npz(reports_genes+"overlap_1.npz",overlap_matrix_1)
# %%
jaccard_matrix_both = get_coef_matrix(jaccard,nodos_enfermedad,both_orders,True)
overlap_matrix_both = get_coef_matrix(overlap,nodos_enfermedad,both_orders,True)

sparse.save_npz(reports_genes+"jaccard_3.npz",jaccard_matrix_both)
sparse.save_npz(reports_genes+"overlap_3.npz",overlap_matrix_both)
#%%

