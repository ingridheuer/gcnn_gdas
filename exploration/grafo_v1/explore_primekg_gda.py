#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import networkx as nx
import igraph as ig 
from matplotlib_venn import venn3, venn3_circles
from matplotlib_venn import venn2
import random
#region load data
#%%
data_path = "../data/interim/"
node_data = pd.read_csv(data_path+"primekg_exploring_node_table.csv", index_col=0)
edge_data = pd.read_csv(data_path+"primekg_exploring_graph_edge_table.csv",index_col=0).rename(columns={"relation":"edge_type"})
seed = 16
random.seed(seed)
#%%
#endregion
def attributes_from_pd(G:nx.Graph,df:pd.DataFrame,attributes:dict):
    for attribute,name in attributes.items():
        nx.set_node_attributes(G,pd.Series(df.set_index("node_index")[attribute]).to_dict(),name)


D = nx.from_pandas_edgelist(edge_data,source="a_index",target="b_index", edge_attr="edge_type")

attributes = {"node_type":"node_type","node_name":"node_name","node_id":"node_id","node_source":"node_source","mondo":"mondo","CUI":"CUI"}

attributes_from_pd(D,node_data,attributes)
#%%
red_enfermedades = edge_data[edge_data.edge_type == "disease_disease"]
enfermedades = node_data[node_data.node_type == "disease"]
dd = nx.from_pandas_edgelist(red_enfermedades,source="a_index",target="b_index",edge_attr="edge_type")

attributes = {"node_type":"node_type","node_name":"node_name","node_id":"node_id","node_source":"node_source","mondo":"mondo","CUI":"CUI"}
attributes_from_pd(dd,node_data,attributes)
#%%
red_gda = edge_data[edge_data.edge_type == "gda"]
gda = nx.from_pandas_edgelist(red_gda,source="a_index",target="b_index",edge_attr="edge_type")
attributes = {"node_type":"node_type","node_name":"node_name","node_id":"node_id","node_source":"node_source","mondo":"mondo","CUI":"CUI"}
attributes_from_pd(gda,node_data,attributes)
#%%
print(f"El grafo completo tiene {D.number_of_nodes()} nodos y {D.number_of_edges()} enlaces")
print(f"La red de enfermedades tiene {dd.number_of_nodes()} nodos y {dd.number_of_edges()} enlaces")
print(f"La red de gen-enfermedad tiene {gda.number_of_nodes()} nodos y {gda.number_of_edges()} enlaces")

#endregion
#region diagrama de venn de las capas
#%%
plt.figure(figsize=[10, 8])
plt.tight_layout()
plt.plot(title='Nodos en comun')

networks = {"red de enfermedades":dd, "red de genes-enfermedades":gda}

(n1, g1), (n2, g2) = networks.items()
s1, s2 = set(g1), set(g2)
'''para dos sets s,t python tiene las siguientes operaciones:
    union: s | t, interseccion: s & t, differencia: s - t
    Se usa esto para definir cuántos elementos hay en cada sección del diagrama'''
center = len(s1 & s2)
venn2([s1,s2], set_labels=[n1, n2])
#%%
#%%
def particion_a_diccionario(Red_igraph,particion_igraph):
  particion_dict = {}
  for cluster in range(len(particion_igraph)):
    for nodo in Red_igraph.vs(particion_igraph[cluster])['name']:
      particion_dict.update({nodo:cluster})
  return particion_dict

def particiones(G):
    dict = {}
    G_igraph = ig.Graph.TupleList(G.edges(), directed=False)
    comunidades_infomap  = G_igraph.community_infomap()
    dict_comunidades_infomap = particion_a_diccionario(G_igraph,comunidades_infomap)
    modularidad_infomap = G_igraph.modularity(comunidades_infomap)
    dict['Infomap'] = {'comunidades' : comunidades_infomap, 'diccionario':dict_comunidades_infomap, 'modularidad':modularidad_infomap}

    #LA FUNCIÓN COMMUNITY_MULTILEVEL DE IGRAPH UTILIZA EL ALGORITMO LOUVAIN
    G_igraph = ig.Graph.TupleList(G.edges(), directed=False)
    comunidades_louvain = G_igraph.community_multilevel()
    dict_comunidades_louvain = particion_a_diccionario(G_igraph,comunidades_louvain)
    modularidad_louvain = G_igraph.modularity(comunidades_louvain)
    dict['Louvain'] = {'comunidades' : comunidades_louvain, 'diccionario':dict_comunidades_louvain, 'modularidad':modularidad_louvain}
    return dict

comunidades_dd = particiones(dd)
#comunidades_ppi = particiones(ppi)
#%%
col_infomap = pd.Series(comunidades_dd['Infomap']['diccionario'],name="comunidades_infomap")
col_louvain = pd.Series(comunidades_dd['Louvain']['diccionario'],name="comunidades_louvain")

enfermedades = enfermedades.join(col_louvain)
enfermedades = enfermedades.join(col_infomap)

nx.set_node_attributes(dd,comunidades_dd['Infomap']['diccionario'],name="comunidad_infomap")
nx.set_node_attributes(dd,comunidades_dd['Louvain']['diccionario'],name="comunidad_louvain")
#%%
#armo un subgrafo para graficarlo
def sample_subgraph(G,porcentaje,nodos=True):
    num_enlaces = G.number_of_edges()
    if nodos:
        num_nodos = G.number_of_nodes()
        k = round((porcentaje*num_nodos)/100)
        sampled_nodes = random.sample(list(G.nodes), k)
        sampled_graph = G.subgraph(sampled_nodes).copy()
    else:
        num_enlaces = G.number_of_edges()
        k = round((porcentaje*num_enlaces)/100)
        sampled_edges = random.sample(list(G.edges),k)
        sampled_graph = G.edge_subgraph(sampled_edges).copy()
    return sampled_graph
#%%
#sample_dd = sample_subgraph(dd,70)
#sample_gda = sample_subgraph(gda,70)
sample_all = sample_subgraph(D,50)
#%%
graphs = {"primekg_sample":sample_all,"primekg_disease_disease":dd,"primekg_gene_disease":gda}

for name,graph in graphs.items():
    nx.write_gml(graph,data_path+name+".gml")

nx.write_gml(D,data_path+"primekg_exploring_full_graph.gml")
