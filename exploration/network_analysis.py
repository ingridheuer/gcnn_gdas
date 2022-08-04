#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import networkx as nx
import igraph as ig 
from matplotlib_venn import venn3, venn3_circles
import random
#region load data
#%%
data_path = "../data/processed/"
node_data = pd.read_csv(data_path+"graph_node_table.csv", index_col=0)
edge_data = pd.read_csv(data_path+"graph_edge_table.csv",index_col=0).rename(columns={"relation":"edge_type"})
seed = 16
random.seed(seed)
#endregion
#%%
#region grafico de enlaces vs year initial y final
red_gda = edge_data[edge_data.edge_type == "GDA"]

nan_count = red_gda.YearInitial.isna().value_counts()
print(f"Hay {nan_count[True]} enlaces sin fecha, para este analisis los saco")

#hay enlaces sin fecha, para este analisis los saco
gda_year_data = red_gda.dropna()

print(f"Me quedo con {gda_year_data.shape[0]} enlaces")
gda_year_data.astype({'YearInitial': 'int','YearFinal':'int'})
#%%
fig_initial = px.histogram(gda_year_data, x="YearInitial", title="Year Initial")
fig_initial.show()
fig_final = px.histogram(gda_year_data, x="YearFinal", title="Year Final")
fig_final.show()

fig_initial_cdf = px.ecdf(gda_year_data, x="YearInitial", title="CDF Year Initial")
fig_initial_cdf.show()

fig_final_cdf = px.ecdf(gda_year_data, x="YearFinal", title="CDF Year Final")
fig_final_cdf.show()
#endregion
#region armo grafos
#%%
def attributes_from_pd(G:nx.Graph,df:pd.DataFrame,attributes:dict):
    for attribute,name in attributes.items():
        nx.set_node_attributes(G,pd.Series(df[attribute], index=df.node_idx).to_dict(),name)


D = nx.from_pandas_edgelist(edge_data,source="a_idx",target="b_idx", edge_attr=["edge_type","YearInitial","YearFinal","score","edge_idx"])

attributes = {"node_type":"node_type","node_name":"node_name","node_idx":"node_dataset_idx","node_id":"node_id","node_source":"node_source","disgenet_type":"disgenet_type","diseaseClassMSH":"diseaseClassMSH","diseaseClassNameMSH":"diseaseClassNameMSH"}

attributes_from_pd(D,node_data,attributes)
#G = D.to_directed()
#%%
red_enfermedades = edge_data[edge_data.edge_type == "parent_child_mondo"]
enfermedades = node_data[node_data.node_type == "disease"]
dd = nx.from_pandas_edgelist(red_enfermedades,source="a_idx",target="b_idx",edge_attr="edge_idx")

attributes = {"node_type":"node_type","node_name":"node_name","node_idx":"node_dataset_idx","node_id":"node_id","node_source":"node_source","disgenet_type":"disgenet_type","diseaseClassMSH":"diseaseClassMSH","diseaseClassNameMSH":"diseaseClassNameMSH"}
attributes_from_pd(dd,node_data,attributes)
#%%
red_proteinas = edge_data[(edge_data.edge_type == "PPI") | (edge_data.edge_type == "forms_complex")]
proteinas = node_data[(node_data.node_type == "gene/protein") | (node_data.node_type == "protein_complex")]
ppi = nx.from_pandas_edgelist(red_proteinas,source="a_idx",target="b_idx",edge_attr="edge_idx")
attributes = {"node_type":"node_type","node_name":"node_name","node_idx":"node_dataset_idx","node_id":"node_id","node_source":"node_source"}
attributes_from_pd(ppi,node_data,attributes)
#%%
gda = nx.from_pandas_edgelist(red_gda,source="a_idx",target="b_idx",edge_attr="edge_idx")
attributes = {"node_type":"node_type","node_name":"node_name","node_idx":"node_dataset_idx","node_id":"node_id","node_source":"node_source","disgenet_type":"disgenet_type","diseaseClassMSH":"diseaseClassMSH","diseaseClassNameMSH":"diseaseClassNameMSH"}
attributes_from_pd(gda,node_data,attributes)

#%%
print(f"El grafo completo tiene {D.number_of_nodes()} nodos y {D.number_of_edges()} enlaces")
print(f"La red de enfermedades tiene {dd.number_of_nodes()} nodos y {dd.number_of_edges()} enlaces")
print(f"La red de proteinas tiene {ppi.number_of_nodes()} nodos y {ppi.number_of_edges()} enlaces")
print(f"La red de gen-enfermedad tiene {gda.number_of_nodes()} nodos y {gda.number_of_edges()} enlaces")

#endregion
#region diagrama de venn de las capas
#%%
plt.figure(figsize=[10, 8])
plt.tight_layout()
plt.plot(title='Nodos en comun')

networks = {"red de enfermedades":dd, "red de proteinas":ppi, "red de genes-enfermedades":gda}

(n1, g1), (n2, g2), (n3, g3) = networks.items()
s1, s2, s3 = set(g1), set(g2), set(g3)
'''para dos sets s,t python tiene las siguientes operaciones:
    union: s | t, interseccion: s & t, differencia: s - t
    Se usa esto para definir cuántos elementos hay en cada sección del diagrama'''
center = len(s1 & s2 & s3)
venn3([len(s1-(s2 | s3)), len(s2-(s1 | s3)), len(s1 & s2)-center,
       len(s3-(s1 | s2)), len(s1 & s3)-center, len(s2 & s3)-center,
       center], set_labels=[n1, n2, n3])

ppi.number_of_nodes()
ppi.number_of_edges()

enfermedades.node_source.value_counts()
enfermedades[enfermedades.node_source == "primekg"]
enfermedades[enfermedades.node_source == "disgenet"]
[data[1]["node_source"] for data in list(dd.nodes(data=True))].count("primekg")
[data[1]["node_source"] for data in list(dd.nodes(data=True))].count("disgenet")
#endregion
#region distribución de grado
#%%
def get_degree_series(G):
    degrees = {n[1]["node_dataset_idx"]:G.degree(n[0]) for n in G.nodes(data=True)}
    return pd.Series(degrees,name="degree")

dd_degree = get_degree_series(dd)
enfermedades = enfermedades.join(dd_degree)

fig_dd_degree = px.histogram(dd_degree, log_y=True, title="Distribución de grado - red de enfermedades")
fig_dd_degree.show()
#%%
ppi_degree = get_degree_series(ppi)
proteinas = proteinas.join(ppi_degree)

fig_ppi_degree = px.histogram(ppi_degree, log_y=True, title="Distribución de grado - red de proteinas")
fig_ppi_degree.show()
#%%
gda_degree = get_degree_series(gda)
nodos_gda = node_data[node_data.node_source == "disgenet"]
nodos_gda = nodos_gda.join(gda_degree)

fig_gda_degree = px.histogram(gda_degree, log_y=True, title="Distribución de grado - red gen enfermedad")
fig_gda_degree.show()
#%%
#Veo algunos nodos
nodos_gda[nodos_gda.degree == max(nodos_gda.degree)]
max_degree_node = nodos_gda.loc[nodos_gda.degree == max(nodos_gda.degree)]["node_idx"].values[0]

enlaces_max_degree = red_gda[(red_gda.a_idx == max_degree_node )|(red_gda.b_idx == max_degree_node)]

idx_nodos_max_degree = pd.unique(enlaces_max_degree.a_idx)
nodos_max_degree = nodos_gda[nodos_gda.node_idx.isin(idx_nodos_max_degree)]
#endregion
#region estudio comunidades
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
#region especificidad genes
enfermedades[enfermedades.disgenet_type == "group"]
nodos_gda[nodos_gda.disgenet_type == "group"]
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
graphs = {"all_types_sample":sample_all,"disease_disease":dd,"protein_protein":ppi,"gene_disease":gda}

for name,graph in graphs.items():
    nx.write_gml(graph,data_path+name+".gml")

nx.write_gml(D,data_path+"full_graph.gml")