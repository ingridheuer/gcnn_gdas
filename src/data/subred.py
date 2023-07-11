#%%
import pandas as pd
import networkx as nx
import os
#%%
data_folder = "../../data/processed/graph_data_nohubs/merged_types/"
node_csv_path = data_folder+"merged_nodes.csv"
edge_csv_path = data_folder+"merged_edges.csv"
save_to_path = data_folder+"subgraph_bauti/"

if not os.path.exists(save_to_path):
    print("save_to_path dir does not exist, a new directory will be created")
    os.makedirs(save_to_path)
#%%
edge_df = pd.read_csv(edge_csv_path,index_col=0)
node_df = pd.read_csv(node_csv_path)
#%%
#me quedo solo con los enlaces gda y ppi
edge_df = edge_df[(edge_df.edge_type == "gda")|(edge_df.edge_type == "ppi")]
nodos_disponibles = edge_df.x_index.unique()
node_df = node_df.set_index("node_index").loc[nodos_disponibles]

edge_df = edge_df[["x_index","y_index","x_type","y_type","edge_type","edge_source"]]
# %%
G_full = nx.from_pandas_edgelist(edge_df,source="x_index",target="y_index")
#%%
#Tomo la componente gigante
Gcc = sorted(nx.connected_components(G_full), key=len, reverse=True)
G = G_full.subgraph(Gcc[0]).copy()
nodos_en_cg = list(G.nodes())
node_df = node_df.loc[nodos_en_cg].sort_values(by="node_index").reset_index()

edge_df = edge_df.set_index("x_index").loc[nodos_en_cg].reset_index()
edge_df = edge_df.set_index("y_index").loc[nodos_en_cg].reset_index()

node_counts = dict(node_df.node_type.value_counts())
node_counts["total"] = sum(node_counts.values())

edge_counts = dict(edge_df.edge_type.value_counts()/2)
edge_counts["total"] = sum(edge_counts.values())

print(pd.DataFrame.from_dict({"Número de nodos":node_counts}, orient="columns"))
print(pd.DataFrame.from_dict({"Número de enlaces":edge_counts}, orient="columns"))

edge_df.to_csv(save_to_path+"edge_data.csv",index=False)
node_df.to_csv(save_to_path+"node_data.csv",index=False)
# %%
def attributes_from_pd(G:nx.Graph,df:pd.DataFrame,attributes:dict,indexcol):
    """Dados un grafo G y un dataframe df con atributos de sus nodos, especificamos los atributos
    que queremos agregar a los nodos en un diccionario con formato {nombre_columna:nombre_atributo}. 
    La función arma un diccionario con los atributos y el nombre que le queremos poner, indexado con el identificador de nodo que elegimos 
    y los asigna a los nodos del grafo"""
    for attribute,name in attributes.items():
        nx.set_node_attributes(G,pd.Series(df.set_index(indexcol)[attribute]).to_dict(),name)

attributes = {"node_type":"node_type","node_name":"node_name","node_id":"node_id","node_source":"node_source"}
attributes_from_pd(G,node_df,attributes,"node_index")
#%%
nx.write_gml(G,save_to_path+"gene_disease_graph.gml")
# %%
