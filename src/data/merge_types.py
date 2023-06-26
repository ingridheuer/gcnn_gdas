#%%
import pandas as pd
data_folder = "../../data/processed/graph_data_nohubs/"

node_df = pd.read_csv(data_folder+"nohub_graph_nodes.csv",index_col="node_index")
edge_df = pd.read_csv(data_folder+"nohub_graph_edge_data.csv")
node_info_df = pd.read_csv(data_folder+"nohub_graph_node_data.csv")

node_df["node_type"] = node_df["node_type"].replace({"bert_group":"disease","complex":"gene_protein"})
node_info_df["node_type"] = node_info_df["node_type"].replace({"bert_group":"disease","complex":"gene_protein"})
edge_df.x_type = edge_df.x_type.replace({"bert_group":"disease","complex":"gene_protein"})
edge_df.y_type = edge_df.y_type.replace({"bert_group":"disease","complex":"gene_protein"})
#%%
node_info_df.to_csv(data_folder+"merged_types/merged_node_info.csv")
node_df.to_csv(data_folder+"merged_types/merged_nodes.csv")
edge_df.to_csv(data_folder+"merged_types/merged_edges.csv")