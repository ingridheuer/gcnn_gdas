# Create a subgraph for demos

#%%
import pandas as pd 
import networkx as nx
#%%
data_processed = "../../data/processed/graph_data_nohubs/"
edge_data = pd.read_csv(data_processed+"nohub_graph_edge_data.csv")
node_data = pd.read_csv(data_processed+"nohub_graph_nodes.csv")

G = nx.from_pandas_edgelist(edge_data,source="x_index",target="y_index")
node = 600
# node = 50
# node = 100
g = nx.ego_graph(G,node,2)
nodelist = list(g.nodes())
print(len(nodelist))
print(g.number_of_edges())
#%%
subgraph_node_data = node_data.set_index("node_index").loc[nodelist]
subgraph_edge_data_temp = edge_data.set_index("x_index").loc[nodelist].reset_index()
subgraph_edge_data = subgraph_edge_data_temp.set_index("y_index").loc[nodelist].reset_index()

subgraph_node_data.to_csv(data_processed+"subgraph_nodes.csv")
subgraph_edge_data.to_csv(data_processed+"subgraph_edges.csv")
#%%
