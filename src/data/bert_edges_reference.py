#%%
import pandas as pd

data_processed = "../../data/processed/"
data_interim = "../../data/interim/"
data_external = "../../data/external/"

disease_mappings = pd.read_csv(data_processed+"graph_data/disease_mappings.csv")
graph_node_data = pd.read_csv(data_processed+"graph_data/grafo_alternativo_CG_nodos.csv")

bert_mask = disease_mappings.group_id_bert.isna()
mapa_nodos_bert = disease_mappings[~bert_mask]
mapa_nodos_no_bert = disease_mappings[bert_mask]
mapa_nodos_bert.sort_values(by="node_index")

nodos_bert = mapa_nodos_bert[["group_id_bert","group_name_bert","node_index"]].drop_duplicates(subset="group_id_bert").reset_index(drop=True)
enlaces_bert_disgenet = mapa_nodos_bert[["diseaseId","group_id_bert"]].drop_duplicates().reset_index(drop=True)

bert_node_data = nodos_bert.rename(columns={"group_id_bert":"node_id","group_name_bert":"node_name","node_index":"prime_node_index"})
bert_node_data["node_type"] = "bert_group"
bert_node_data["node_source"] = "primekg"

bert_edge_data = enlaces_bert_disgenet.rename(columns={"diseaseId":"x_id","group_id_bert":"y_id"})
bert_edge_data["relation"] = "disease_bert"
bert_edge_data["x_type"] = "disease"
bert_edge_data["y_type"] = "bert_group"
bert_edge_data["edge_source"] = "primekg"

bert_temp = pd.merge(bert_edge_data, graph_node_data[["node_index","node_id"]], left_on="x_id", right_on="node_id",how="left").rename(columns={"node_index":"x_index"}).drop(columns=["node_id"]).dropna().astype({"x_index":"int"})

bert_final = pd.merge(bert_temp, graph_node_data[["node_index","node_id"]], left_on="y_id", right_on="node_id",how="left").rename(columns={"node_index":"y_index"}).drop(columns=["node_id"]).dropna().astype({"y_index":"int"})

bert_final.to_csv(data_processed+"graph_data/bert_edge_reference.csv")