#%%
import numpy as np
import pandas as pd
import networkx as nx
import scipy.stats as stats
#%%
data_processed = "../../data/processed/"
data_interim = "../../data/interim/"
data_external = "../../data/external/"
graph_data = data_processed + "graph_data_nohubs/"

reports = "../../reports/reports_nohubs/"

graph_node_data = pd.read_csv(graph_data+"nohub_graph_node_data.csv")
graph_edge_data = pd.read_csv(graph_data+"nohub_graph_edge_data.csv")

GDA = nx.read_gml(graph_data+"nohub_gda_network.gml", destringizer=int)
#%%
def get_cluster_nodesets(graph_node_data,series=True):
    infomap_sets = graph_node_data[["node_index","comunidades_infomap"]].dropna().astype({"comunidades_infomap":int}).groupby("comunidades_infomap")["node_index"].apply(set)

    louvain_sets = graph_node_data[["node_index","comunidades_louvain"]].dropna().astype({"comunidades_louvain":int}).groupby("comunidades_louvain")["node_index"].apply(set)

    if series:
        return infomap_sets, louvain_sets
    else:
        return infomap_sets.to_dict(), louvain_sets.to_dict()

def neighbors_from_list(node_list:list,G):
    neighbor_lists = [G.neighbors(n) for n in node_list] #list of lists
    unnested_set = {item for sublist in neighbor_lists for item in sublist} #set of nodes in lists
    return unnested_set

def get_node_neighbor_sets(node_list:list,G):
    node_sets = {node:set(G.neighbors(node)) for node in node_list}
    return node_sets

def get_cluster_neighbor_sets(set_nodos_enfermedad:set,cluster_nodesets:dict,G):
    nodos_en_gda = {cluster:list(nodeset&set_nodos_enfermedad) for cluster,nodeset in cluster_nodesets.items()}
    cluster_gene_neighbors = {cluster:neighbors_from_list(nodelist,G) for cluster,nodelist in nodos_en_gda.items()}
    return cluster_gene_neighbors

def attributes_from_pd(G:nx.Graph,df:pd.DataFrame,attributes:dict,indexcol):
    """Dados un grafo G y un dataframe df con atributos de sus nodos, especificamos los atributos
    que queremos agregar a los nodos en un diccionario con formato {nombre_columna:nombre_atributo}. 
    La función arma un diccionario con los atributos y el nombre que le queremos poner, indexado con el identificador de nodo que elegimos 
    y los asigna a los nodos del grafo"""
    for attribute,name in attributes.items():
        nx.set_node_attributes(G,pd.Series(df.set_index(indexcol)[attribute]).to_dict(),name)

def build_protein_pathway_graph(graph_edge_data):
    protein_gene_edges = graph_edge_data[graph_edge_data.edge_type == "pathway_protein"]
    PP = nx.from_pandas_edgelist(protein_gene_edges, source="x_index",target="y_index")
    attributes_from_pd(PP, graph_node_data, {"node_name":"node_name","node_type":"node_type","node_id":"node_id"},"node_index")
    return PP

def get_pathway_sets(PP,pathway_nodes):
    pathway_sets = {}
    for pathway in pathway_nodes:
        vecinos = set(PP.neighbors(pathway))
        pathway_sets[pathway] = vecinos
    
    return pathway_sets

#%%
nodos_gda = pd.DataFrame(dict(GDA.nodes(data=True))).T.reset_index().rename(columns={"index":"node_index"})
set_nodos_enfermedad = set(nodos_gda.loc[nodos_gda.node_type == "disease", "node_index"].sort_values().values)

all_genes = set(graph_node_data.loc[graph_node_data.node_type == "gene_protein", "node_index"].values)
pathway_nodes = graph_node_data.loc[graph_node_data.node_type == "pathway","node_index"]
#%%
infomap_nodesets, louvain_nodesets = get_cluster_nodesets(graph_node_data,series=False)
infomap_sets = get_cluster_neighbor_sets(set_nodos_enfermedad,infomap_nodesets,GDA)
louvain_sets = get_cluster_neighbor_sets(set_nodos_enfermedad,louvain_nodesets,GDA)

PP = build_protein_pathway_graph(graph_edge_data)
pathway_sets = get_pathway_sets(PP,pathway_nodes)
#%%
def fisher_test(A,B,all_genes):
    intersection = A&B
    test = False
    odds_ratio = 0
    pvalue = 0

    if len(intersection) != 0:
        test = True
        A_barra = all_genes - A 
        B_barra = all_genes - B
        matrix = [[len(intersection), len(A_barra&B)], [len(A&B_barra), len(A_barra&B_barra)]]
        odds_ratio, pvalue = stats.fisher_exact(matrix)
    
    return test, (odds_ratio,pvalue)

def test_all_clusters(cluster_sets,pathway_sets,all_genes):
    results = {}
    total_clusters = len(cluster_sets)
    i = 0
    for cluster,c_set in cluster_sets.items():
        print(f"Testing cluster {i+1} of {total_clusters}")
        # partial = {}
        for pathway, p_set in pathway_sets.items():
            test, (odds_ratio,pvalue) = fisher_test(c_set,p_set,all_genes)
            if test:
                results[(cluster,pathway)] = {"odds_ratio":odds_ratio, "pvalue":pvalue}

        # results[cluster] = partial
        i += 1

    print("Done")
    return results
#%%
louvain_results = test_all_clusters(louvain_sets,pathway_sets,all_genes)
infomap_results = test_all_clusters(infomap_sets,pathway_sets,all_genes)
#%%
louvain_results_df = pd.DataFrame(louvain_results).T
louvain_results_df.index = louvain_results_df.index.set_names(["cluster","pathway"])

infomap_results_df = pd.DataFrame(infomap_results).T
infomap_results_df.index = infomap_results_df.index.set_names(["cluster","pathway"])
#%%
infomap_results_df.to_csv(reports+"analisis_red_genes/infomap_pathways.csv")
louvain_results_df.to_csv(reports+"analisis_red_genes/louvain_pathways.csv")
#%%