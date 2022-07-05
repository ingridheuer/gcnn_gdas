#%%
#from hsage_dotprod import HeteroGNN, train, test
import networkx as nx
import matplotlib.pyplot as plt
import torch
from deepsnap.hetero_graph import HeteroGraph
from deepsnap.dataset import GraphDataset
import copy
from torch_sparse import SparseTensor
from torch_geometric.transforms import ToSparseTensor
from torch.utils.tensorboard import SummaryWriter
#region make graph
#%%
# Create small CPU friendly hetero dataset from zacahary's karate club network
# First initialize de nx.Graph object and assign node and edge types
plotting = True
G = nx.karate_club_graph()
community_map = {}
for node in G.nodes(data=True):
    if node[1]["club"] == "Mr. Hi":
        community_map[node[0]] = 0
    else:
        community_map[node[0]] = 1
    node_color = []
color_map = {0: 0, 1: 1}
node_color = [color_map[community_map[node]] for node in G.nodes()]
pos = nx.spring_layout(G)

if plotting:
    plt.figure(figsize=(7, 7))
    plt.title(
        "Nodos coloreados por node type (pertenencia a club 'Mr. Hi' o 'Officer')")
    nx.draw(G, pos=pos, cmap=plt.get_cmap('coolwarm'), node_color=node_color)
# %%
def assign_node_types(G, community_map):
    # function that takes in a NetworkX graph
    # G and community map assignment (mapping node id --> 0/1 label)
    # and adds 'node_type' as a node_attribute in G.

    nodetype_map = {n: ['n0', 'n1'][community_map[n]]
                    for n in community_map.keys()}
    nx.set_node_attributes(G, nodetype_map, 'node_type')


def assign_node_labels(G, community_map):
    # function that takes in a NetworkX graph
    # G and community map assignment (mapping node id --> 0/1 label)
    # and adds 'node_label' as a node_attribute in G.
    node_labels = community_map
    nx.set_node_attributes(G, node_labels, 'node_label')


def assign_node_features(G):
    # function that takes in a NetworkX graph
    # G and adds 'node_feature' as a node_attribute in G. Each node
    # in the graph has the same feature vector - a torchtensor with
    # data [1., 1., 1., 1., 1.]
    node_feature = torch.ones(5)
    nx.set_node_attributes(G, node_feature, 'node_feature')


assign_node_types(G, community_map)
assign_node_labels(G, community_map)
assign_node_features(G)

# Explore node properties for the node with id: 20
node_id = 5
print(f"Node {node_id} has properties:", G.nodes(data=True)[node_id])
# %%


def assign_edge_types(G, community_map):
    # function that takes in a NetworkX graph
    # G and community map assignment (mapping node id --> 0/1 label)
    # and adds 'edge_type' as a edge_attribute in G.

    edges = list(G.edges())
    edgetype_rule = {(0, 0): 'e0', (1, 1): 'e1', (1, 0): 'e2', (0, 1): 'e2'}
    # {enlace : tipo enlace[ comunidad nodo 1, comunidad nodo 2]}
    edgetype_map = {
        e: edgetype_rule[community_map[e[0]], community_map[e[1]]] for e in edges}
    nx.set_edge_attributes(G, edgetype_map, 'edge_type')


assign_edge_types(G, community_map)
# Explore edge properties for a sampled edge and check the corresponding
# node types
edges = list(G.edges())
edge_idx = 25
n1 = edges[edge_idx][0]
n2 = edges[edge_idx][1]
edge = list(G.edges(data=True))[edge_idx]
print(f"Edge ({edge[0]}, {edge[1]}) has properties:", edge[2])
print(f"Node {n1} has properties:", G.nodes(data=True)[n1])
print(f"Node {n2} has properties:", G.nodes(data=True)[n2])
# %%
# Now initialize a deepsnap HeteroGraph object from the Graph G and explore it

hete = HeteroGraph(G)

def get_nodes_per_type(hete):
    # function that takes a DeepSNAP dataset object
    # and return the number of nodes per `node_type`.
    num_nodes_n0 = hete.num_nodes()['n0']
    num_nodes_n1 = hete.num_nodes()['n1']

    return num_nodes_n0, num_nodes_n1


num_nodes_n0, num_nodes_n1 = get_nodes_per_type(hete)
print("Node type n0 has {} nodes".format(num_nodes_n0))
print("Node type n1 has {} nodes".format(num_nodes_n1))

if plotting:
    edge_color = {}
    for edge in G.edges():
        n1, n2 = edge
        edge_color[edge] = community_map[n1] if community_map[n1] == community_map[n2] else 2
        if community_map[n1] == community_map[n2] and community_map[n1] == 0:
            edge_color[edge] = 'blue'
        elif community_map[n1] == community_map[n2] and community_map[n1] == 1:
            edge_color[edge] = 'red'
        else:
            edge_color[edge] = 'green'

    G_orig = copy.deepcopy(G)
    nx.classes.function.set_edge_attributes(G, edge_color, name='color')
    colors = nx.get_edge_attributes(G, 'color').values()
    labels = nx.get_node_attributes(G, 'node_type')
    plt.figure(figsize=(8, 8))
    plt.title("Nodos y enlaces coloreados por tipo")
    nx.draw(G, pos=pos, cmap=plt.get_cmap('coolwarm'), node_color=node_color,
            edge_color=colors, labels=labels, font_color='white')


def get_num_message_edges(hete):
    # function that takes a DeepSNAP dataset object
    # and return the number of edges for each message type.
    # You should return a list of tuples as
    # (message_type, num_edge)

    message_type_edges = []
    edgetype_dict = hete.num_edges()
    message_type_edges = [(t[1], n) for (t, n) in edgetype_dict.items()]
    return message_type_edges


message_type_edges = get_num_message_edges(hete)
for (message_type, num_edges) in message_type_edges:
    print("Message type {} has {} edges".format(message_type, num_edges))

#endregion
# %%
#region old split
# Make the hetero dataset splits
# Modify it for edge prediction task in transductive setting

# def compute_dataset_split_counts(datasets):
#     # function that takes a dict of datasets in the form
#     # {'train': dataset_train, 'val': dataset_val, 'test': dataset_test}
#     # and returns a dict mapping dataset names to the number of labeled
#     # nodes used for supervision in that respective dataset.

#     data_set_splits = {}
#     data_set_splits['train'] = sum(
#         [len(v) for v in datasets['train'][0].node_label_index.values()])
#     data_set_splits['val'] = sum(
#         [len(v) for v in datasets['val'][0].node_label_index.values()])
#     data_set_splits['test'] = sum(
#         [len(v) for v in datasets['test'][0].node_label_index.values()])

#     return data_set_splits


# dataset = GraphDataset([hete], task='link_pred')
# # Splitting the dataset
# dataset_train, dataset_val, dataset_test = dataset.split(
#     transductive=True, split_ratio=[0.4, 0.3, 0.3])
# datasets = {'train': dataset_train, 'val': dataset_val, 'test': dataset_test}

# data_set_splits = compute_dataset_split_counts(datasets)
# for dataset_name, num_nodes in data_set_splits.items():
#     print("{} dataset has {} nodes".format(dataset_name, num_nodes))

# titles = ['Train', 'Validation', 'Test']

# for i, dataset in enumerate([dataset_train, dataset_val, dataset_test]):
#     n0 = hete._convert_to_graph_index(
#         dataset[0].node_label_index['n0'], 'n0').tolist()
#     n1 = hete._convert_to_graph_index(
#         dataset[0].node_label_index['n1'], 'n1').tolist()

#     if plotting:
#         plt.figure(figsize=(7, 7))
#         plt.title(titles[i])
#         nx.draw(G_orig, pos=pos, node_color="grey",
#                 edge_color=colors, labels=labels, font_color='white')
#         nx.draw_networkx_nodes(G_orig.subgraph(n0), pos=pos, node_color="blue")
#         nx.draw_networkx_nodes(G_orig.subgraph(n1), pos=pos, node_color="red")
# endregion

#region new split
#%%
G = nx.karate_club_graph()
assign_edge_types(G,community_map)
assign_node_types(G,community_map)
assign_node_features(G)

if plotting:
    pos = nx.spring_layout(G, seed=1)
    plt.figure(figsize=(8, 8))
    plt.title("Karate club, nodos coloreados por pertenencia a club")
    nx.draw(G, pos=pos, cmap=plt.get_cmap('coolwarm'), connectionstyle='arc3, rad = 0.1', edge_color='grey', labels=labels, node_color=node_color)

# edges = list(G.edges())
# edge_idx = 24
# n1 = edges[edge_idx][0]
# n2 = edges[edge_idx][1]
# edge = list(G.edges(data=True))[edge_idx]
# print(f"Edge ({edge[0]}, {edge[1]}) has properties:", edge[2])
# print(f"Node {n1} has properties:", G.nodes(data=True)[n1])
# print(f"Node {n2} has properties:", G.nodes(data=True)[n2])

#%%
#Transformo el objeto de nx a un deepsnap dataset heterogeneo
if plotting:
    task = 'link_pred'
    dg = HeteroGraph(G)

    dataset = GraphDataset([dg], task=task, edge_train_mode="all")
    dataset_train, dataset_val, dataset_test = dataset.split(transductive=True, split_ratio=[0.4, 0.3, 0.3])
    titles = ['Train', 'Validation', 'Test']
    edge_types = [('n0','e0','n0'),('n0','e2','n1'),('n1','e1','n1')]

#%%
#Visualizo como quedan los splits}
if plotting:
    mode = dataset.edge_train_mode
    for i, dataset in enumerate([dataset_train, dataset_val, dataset_test]):
        edge_color_dataset = {}
        for edge in G.edges():
            n1, n2 = edge
            edge_color_dataset[edge] = "grey"
        for message_type in dataset[0].edge_label_index:
            if message_type[0] == 'n0' and message_type[2] == 'n0':
                from_nodes = dg._convert_to_graph_index(dataset[0].edge_label_index[('n0', 'e0', 'n0')][0], 'n0')
                to_nodes = dg._convert_to_graph_index(dataset[0].edge_label_index[('n0', 'e0', 'n0')][1], 'n0')
            elif message_type[0] == 'n1' and message_type[2] == 'n1':
                from_nodes = dg._convert_to_graph_index(dataset[0].edge_label_index[('n1', 'e1', 'n1')][0], 'n1')
                to_nodes = dg._convert_to_graph_index(dataset[0].edge_label_index[('n1', 'e1', 'n1')][1], 'n1')
            else:
                from_nodes = dg._convert_to_graph_index(dataset[0].edge_label_index[('n0', 'e2', 'n1')][0], 'n0')
                to_nodes = dg._convert_to_graph_index(dataset[0].edge_label_index[('n0', 'e2', 'n1')][1], 'n1')
            for j in range(len(from_nodes)):
                edge = (from_nodes[j].item(), to_nodes[j].item())
                if G.has_edge(edge[0], edge[1]):
                    if message_type[0] == 'n0' and message_type[2] == 'n0':
                        edge_color_dataset[edge] = 'blue'
                    elif message_type[0] == 'n1' and message_type[2] == 'n1':
                        edge_color_dataset[edge] = 'red'
                    else:
                        edge_color_dataset[edge] = 'green'
        H = copy.deepcopy(G)
        nx.classes.function.set_edge_attributes(H, edge_color_dataset, name='color_dataset')
        edge_color_dataset = nx.get_edge_attributes(H, 'color_dataset').values()
        plt.figure(figsize=(7, 7))
        plt.title(f'{titles[i]} supervision messages. Train mode: {mode}', fontsize=20)
        nx.draw(H, pos=pos, node_color="grey", edge_color=edge_color_dataset, labels=labels, font_color='white')
        plt.show()
print(f"Edgetypes: {edge_types}")
#%%

#quiero visualizar como se dividen los edges mensajeros y los de supervisión
if plotting:
    for i, dataset in enumerate([dataset_train,dataset_val,dataset_test]):
        edge_color_dataset = {}
        for edge in G.edges():
            n1,n2 = edge 
            edge_color_dataset[edge] = "grey"
        for message_type in dataset[0].edge_index:
            if message_type == edge_types[0]:
                from_nodes = dg._convert_to_graph_index(dataset[0].edge_index[edge_types[0]][0], edge_types[0][0])
                to_nodes = dg._convert_to_graph_index(dataset[0].edge_index[edge_types[0]][1], edge_types[0][2])
            elif message_type == edge_types[1]:
                from_nodes = dg._convert_to_graph_index(dataset[0].edge_index[edge_types[1]][0], edge_types[1][0])
                to_nodes = dg._convert_to_graph_index(dataset[0].edge_index[edge_types[1]][1], edge_types[1][2])
            else:
                from_nodes = dg._convert_to_graph_index(dataset[0].edge_index[edge_types[2]][0], edge_types[2][0])
                to_nodes = dg._convert_to_graph_index(dataset[0].edge_index[edge_types[2]][1], edge_types[2][2])
            for j in range(len(from_nodes)):
                edge = (from_nodes[j].item(), to_nodes[j].item())
                if G.has_edge(edge[0], edge[1]):
                    if message_type == edge_types[0]:
                        edge_color_dataset[edge] = 'blue'
                    elif message_type == edge_types[2]:
                        edge_color_dataset[edge] = 'red'
                    else:
                        edge_color_dataset[edge] = 'green'
        H = copy.deepcopy(G)
        nx.classes.function.set_edge_attributes(H, edge_color_dataset, name='color_dataset')
        edge_color_dataset = nx.get_edge_attributes(H, 'color_dataset').values()
        plt.figure(figsize=(7, 7))
        plt.title(f'{titles[i]} message edges. Train mode {mode}', fontsize=20)
        nx.draw(H, pos=pos, node_color="grey", edge_color=edge_color_dataset, labels=labels, font_color='white')
        plt.show()
print(f"Edgetypes: {edge_types}")
#endregion
#%%
#region try to train

args = {
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'hidden_size': 64,
    'epochs': 100,
    'weight_decay': 1e-5,
    'lr': 0.003,
    'attn_size': 32,
}

task = 'link_pred'
dg = HeteroGraph(G)

# # Edge_index to sparse tensor and to device
# for key in dg.edge_index:
#     edge_index = dg.edge_index[key]
#     from_type = key[0]
#     to_type = key[2]
#     adj = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(dg.num_nodes(from_type), dg.num_nodes(to_type)))
#     dg.edge_index[key] = adj.t()

# for key in dg.edge_label_index:
#     edge_index = dg.edge_label_index[key]
#     from_type = key[0]
#     to_type = key[2]
#     adj = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(dg.num_nodes(from_type), dg.num_nodes(to_type)))
#     dg.edge_label_index[key] = adj.t()


dataset = GraphDataset([dg], task=task, edge_train_mode="disjoint")
datasets = dataset_train, dataset_val, dataset_test = dataset.split(transductive=True, split_ratio=[0.4, 0.3, 0.3])
dataset_sparse = {}

for dataset_set in datasets:
    #transformo los edge index a sparse tensors
    for key, index in dataset_set[0].edge_index.items():
        from_type = key[0]
        to_type = key[2]
        print(f'key = {key}, index = {index}')
        adj = SparseTensor(row=index[0], col=index[1], sparse_sizes=(dataset_set[0].num_nodes(from_type), dataset_set[0].num_nodes(to_type)))
        dataset_set[0].edge_index[key] = adj

model = HeteroGNN(dataset[0], args, aggr="mean")
optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

train(model,optimizer,dataset_train[0])
