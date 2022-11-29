#%%
import networkx as nx
import pandas as pd
#%%
graph_node_data = pd.read_csv("../../data/processed/graph_data_nohubs/nohub_graph_node_data.csv")
graph_edge_data = pd.read_csv("../../data/processed/graph_data_nohubs/nohub_graph_edge_data.csv")
#%%
def attributes_from_pd(G:nx.Graph,df:pd.DataFrame,attributes:dict,indexcol):
    """Dados un grafo G y un dataframe df con atributos de sus nodos, especificamos los atributos
    que queremos agregar a los nodos en un diccionario con formato {nombre_columna:nombre_atributo}. 
    La función arma un diccionario con los atributos y el nombre que le queremos poner, indexado con el identificador de nodo que elegimos 
    y los asigna a los nodos del grafo"""
    for attribute,name in attributes.items():
        nx.set_node_attributes(G,pd.Series(df.set_index(indexcol)[attribute]).to_dict(),name)
#%%
G = nx.from_pandas_edgelist(graph_edge_data,source="x_index",target="y_index", edge_attr="edge_type")

G_attributes = {"node_name":"node_name","node_type":"node_type","node_id":"node_id"}

attributes_from_pd(G,graph_node_data,G_attributes,"node_index")
#%%
nx.write_gml(G,"../../data/processed/graph_data_nohubs/processed_graph.gml")