#%%
import pandas as pd

data_processed = "../../data/processed/"
data_interim = "../../data/interim/"
data_external = "../../data/external/"

disease_mappings = pd.read_csv(data_processed+"graph_data_nohubs/disease_mappings.csv")
graph_node_data = pd.read_csv(data_processed+"graph_data_nohubs/nohub_graph_nodes.csv")
graph_edge_data = pd.read_csv(data_processed+"graph_data_nohubs/nohub_graph_edge_data.csv")

bert_mask = disease_mappings.group_id_bert.isna()
mapa_nodos_bert = disease_mappings[~bert_mask]
mapa_nodos_no_bert = disease_mappings[bert_mask]
mapa_nodos_bert.sort_values(by="prime_node_index")

nodos_bert = mapa_nodos_bert[["group_id_bert","group_name_bert","prime_node_index"]].drop_duplicates(subset="group_id_bert").reset_index(drop=True)
enlaces_bert_disgenet = mapa_nodos_bert[["diseaseId","group_id_bert"]].drop_duplicates().reset_index(drop=True)

bert_node_data = nodos_bert.rename(columns={"group_id_bert":"node_id","group_name_bert":"node_name"})
bert_node_data["node_type"] = "bert_group"
bert_node_data["node_source"] = "primekg"

bert_edge_data = enlaces_bert_disgenet.rename(columns={"diseaseId":"x_id","group_id_bert":"y_id"})
bert_edge_data["relation"] = "disease_bert_map"
bert_edge_data["x_type"] = "disease"
bert_edge_data["y_type"] = "bert_group"
bert_edge_data["edge_source"] = "primekg"

bert_temp = pd.merge(bert_edge_data, graph_node_data[["node_index","node_id"]], left_on="x_id", right_on="node_id",how="left").rename(columns={"node_index":"x_index"}).drop(columns=["node_id"]).dropna().astype({"x_index":"int"})

bert_edges = pd.merge(bert_temp, graph_node_data[["node_index","node_id"]], left_on="y_id", right_on="node_id",how="left").rename(columns={"node_index":"y_index"}).drop(columns=["node_id"]).dropna().astype({"y_index":"int"})

dd = graph_edge_data[graph_edge_data.edge_type == "disease_disease"]

temp_df = pd.merge(dd,bert_edges[["x_index","y_index","relation"]],left_on=["x_index","y_index"], right_on=["x_index","y_index"], how="inner").rename(columns={"relation":"disease_edge_type"})

bert_edges_inverted = bert_edges.rename(columns={"x_index":"y_index","y_index":"x_index"})

temp_df_inverted = pd.merge(dd,bert_edges_inverted[["x_index","y_index","relation"]],left_on=["x_index","y_index"], right_on=["x_index","y_index"], how="inner").rename(columns={"relation":"disease_edge_type"})

final_bert_edges = pd.concat([temp_df,temp_df_inverted])
final_bert_edges.to_csv(data_processed+"graph_data_nohubs/disease_bert_edges.csv")