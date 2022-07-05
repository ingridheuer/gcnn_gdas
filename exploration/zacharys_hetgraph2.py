#%%
from sympy import true
import torch
from torch_sparse import matmul
from torch_sparse import SparseTensor
import torch_geometric.nn as pyg_nn
from deepsnap.dataset import GraphDataset
import deepsnap.hetero_gnn
from deepsnap.hetero_gnn import forward_op
from deepsnap.hetero_graph import HeteroGraph
from torch_geometric.nn import HeteroConv
from deepsnap.hetero_gnn import HeteroSAGEConv
from torch.utils.data import DataLoader
from deepsnap.batch import Batch
#import torch_geometric.transforms as T
from timeit import default_timer as timer
from torch_geometric.data import HeteroData
import numpy as np
#import deepsnap
import torch.nn as nn
#import torch.nn.functional as F
from sklearn.metrics import f1_score

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import copy
from torch.utils.tensorboard import SummaryWriter
# %%
debugging = False

def my_debug(arg):
    if debugging:
        print(arg)
# %%
# region MODEL CLASS DEFINITION (GRAPH STYLE)

class HeteroGNNConv(pyg_nn.MessagePassing):
    def __init__(self, in_channels_src, in_channels_dst, out_channels):
        super().__init__(aggr="mean")

        self.in_channels_src = in_channels_src
        self.in_channels_dst = in_channels_dst
        self.out_channels = out_channels
        self.lin_dst = torch.nn.Linear(in_channels_dst, out_channels)
        self.lin_src = torch.nn.Linear(in_channels_src, out_channels)
        self.lin_update = torch.nn.Linear(2*out_channels, out_channels)

    def forward(
            self,
            node_feature_src,
            node_feature_dst,
            edge_index,
            size=None):

        out = self.propagate(
            edge_index, size, node_feature_src=node_feature_src, node_feature_dst=node_feature_dst)
        return out

    def message_and_aggregate(self, edge_index, node_feature_src):
        out = matmul(edge_index, node_feature_src, reduce=self.aggr)
        #my_debug(f'message_and_agregate out: {out}')

        return out

    def update(self, aggr_out, node_feature_dst):

        dst_msg = self.lin_dst(node_feature_dst)
        src_msg = self.lin_src(aggr_out)
        full_msg = torch.concat((dst_msg, src_msg), dim=1)
        out = self.lin_update(full_msg)
        #my_debug(f'update out: {out}')
        return out


class HeteroGNNWrapperConv(deepsnap.hetero_gnn.HeteroConv):
    def __init__(self, convs, args, aggr="mean"):
        super().__init__(convs, None)
        self.aggr = aggr

        # Map the index and message type
        self.mapping = {}

    def reset_parameters(self):
        super().reset_parameters()

    def forward(self, node_features, edge_indices):
        message_type_emb = {}
        #my_debug(f'Wrapper self convs: {self.convs}')
        for message_key, message_type in edge_indices.items():
            src_type, edge_type, dst_type = message_key
            # my_debug(message_key)
            # my_debug(node_features.shape)
            #my_debug(f"Input x shape: {node_features.shape}")
            #my_debug(f'Wrapper self conv step: {self.convs[message_key]}')
            node_feature_src = node_features[src_type]
            node_feature_dst = node_features[dst_type]
            edge_index = edge_indices[message_key]
            #my_debug(f'Wrapper message_type_emb step: {self.convs[message_key](node_feature_src,node_feature_dst,edge_index)}')
            message_type_emb[message_key] = (
                self.convs[message_key](
                    node_feature_src,
                    node_feature_dst,
                    edge_index,
                )
            )
            #my_debug(f'{message_key} emb done: {message_type_emb}')
        node_emb = {dst: [] for _, _, dst in message_type_emb.keys()}
        #my_debug(f'node emb: {node_emb}')
        mapping = {}
        for (src, edge_type, dst), item in message_type_emb.items():
            mapping[len(node_emb[dst])] = (src, edge_type, dst)
            node_emb[dst].append(item)
        self.mapping = mapping
        for node_type, embs in node_emb.items():
            if len(embs) == 1:
                node_emb[node_type] = embs[0]
            else:
                node_emb[node_type] = self.aggregate(embs)
        #my_debug(f'node_emb from wrapper conv forward:{node_emb}')
        return node_emb

    def aggregate(self, xs):
        return torch.mean(torch.stack(xs), dim=0)


def generate_convs(hetero_graph, conv, hidden_size, first_layer=False):
    convs = {}

    msg_types = hetero_graph.message_types
    for key in msg_types:
        if first_layer:
            dst_feature_dim = hetero_graph.num_node_features(key[2])
            src_feature_dim = hetero_graph.num_node_features(key[0])
            convs[key] = conv(src_feature_dim, dst_feature_dim, hidden_size)
        else:
            convs[key] = conv(hidden_size, hidden_size, hidden_size)

    return convs


class HeteroGNN(torch.nn.Module):
    def __init__(self, hetero_graph, args, aggr="mean"):
        super().__init__()

        self.aggr = aggr
        self.hidden_size = args['hidden_size']
        self.bns1 = torch.nn.ModuleDict()
        self.bns2 = torch.nn.ModuleDict()
        self.relus1 = torch.nn.ModuleDict()
        self.relus2 = torch.nn.ModuleDict()
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        #self.post_mps = nn.ModuleDict()

        convs1 = generate_convs(hetero_graph, HeteroGNNConv, self.hidden_size, first_layer=True)
        #my_debug(f'convs1: {convs1}')
        convs2 = generate_convs(hetero_graph, HeteroGNNConv, self.hidden_size, first_layer=False)
        self.convs1 = HeteroGNNWrapperConv(convs1, args, aggr=self.aggr)
        #my_debug(f'self.convs1: {self.convs1}')
        self.convs2 = HeteroGNNWrapperConv(convs2, args, aggr=self.aggr)
        for node_type in hetero_graph.node_types:
            self.bns1[node_type] = torch.nn.BatchNorm1d(self.hidden_size)
            self.bns2[node_type] = torch.nn.BatchNorm1d(self.hidden_size)
            self.relus1[node_type] = torch.nn.LeakyReLU()
            self.relus2[node_type] = torch.nn.LeakyReLU()

    def forward(self, data):
        x, adj, edge_label_index = data.node_feature, data.sparse_mat, data.edge_label_index
        x = self.convs1(x, edge_indices=adj)
        x = deepsnap.hetero_gnn.forward_op(x, self.bns1)
        x = deepsnap.hetero_gnn.forward_op(x, self.relus1)
        x = self.convs2(x, edge_indices=adj)
        x = deepsnap.hetero_gnn.forward_op(x, self.bns2)
        #my_debug(f"Output x shape: {x.shape}")
        x = deepsnap.hetero_gnn.forward_op(x, self.relus2)
        #x = forward_op(x, self.post_mps)

        pred = {}
        for message_type in edge_label_index:
            my_debug(f"{len(edge_label_index[message_type][0])}")
            src_type = message_type[0]
            trg_type = message_type[2]
            nodes_first = torch.index_select(x[src_type], 0, edge_label_index[message_type][0,:].long())
            nodes_second = torch.index_select(x[trg_type], 0, edge_label_index[message_type][1,:].long())
            my_debug(f"Multiplying shapes {nodes_first.shape}, {nodes_second.shape}")
            pred[message_type] = torch.sum(nodes_first * nodes_second, dim=-1)
            my_debug(f"Pred shape {pred[message_type].shape}")
        return pred

    def loss(self, pred, y):
        loss = 0
        for key in pred:
            p = torch.sigmoid(pred[key])
            loss += self.loss_fn(p, y[key].type(pred[key].dtype))
        return loss
# endregion
# %%
# region DEF TRAIN AND TEST FUNCTIONS GRAPH VER

def train(model, optimizer, dataset, printb):
    model.train()
    optimizer.zero_grad()
    preds = model(dataset[0])
    loss = model.loss(preds, dataset[0].edge_label)
    loss.backward()
    optimizer.step()
    if printb:
        print(loss.item())
    return loss.item()

# Test function
def test(model, graph_dict, args):
    model.eval()
    accs = {}
    for mode, dataset in graph_dict.items():
        acc = 0
        num = 0
        pred = model(dataset[0])
        for key in pred:
            p = torch.sigmoid(pred[key]).cpu().detach().numpy()
            pred_label = np.zeros_like(p, dtype=np.int64)
            pred_label[np.where(p > 0.5)[0]] = 1
            pred_label[np.where(p <= 0.5)[0]] = 0
            acc += np.sum(pred_label == dataset[0].edge_label[key].cpu().numpy())
            num += len(pred_label)
        accs[mode] = acc / num
    return accs
# endregion
#%%
# region build graph
G = nx.karate_club_graph()
community_map = {}
for node in G.nodes(data=True):
    if node[1]["club"] == "Mr. Hi":
        community_map[node[0]] = 0
    else:
        community_map[node[0]] = 1

#asigno tipos de nodo según comunidad
nodetype_map = {n: ['n0', 'n1'][community_map[n]] for n in community_map.keys()}

nx.set_node_attributes(G, nodetype_map, 'node_type')
nx.set_node_attributes(G, nodetype_map, 'node_label')

#inicializo las features como un vector de unos
node_feature = torch.ones(5)
nx.set_node_attributes(G, node_feature, 'node_feature')

#asigno tipos de enlaces
edgetype_rule = {(0, 0): 'e0', (1, 1): 'e1', (1, 0): 'e2', (0, 1): 'e2'}
edges = list(G.edges())
# {enlace : tipo enlace[ comunidad nodo 1, comunidad nodo 2]}
edgetype_map = {e: edgetype_rule[community_map[e[0]], community_map[e[1]]] for e in edges}
nx.set_edge_attributes(G, edgetype_map, 'edge_type')

hete = HeteroGraph(G)
#endregion
#%%
#region split dataset
#extra pre-processing to make data compatible with model, for GraphSAGE we need to input edge indices as sparse matrices, but deepsnap works with tensors. We're going to pass a dictionary as argument to the model
def edgeindex_to_sparsematrix(het_graph: HeteroGraph): 
    sparse_edge_dict = {}
    for key in het_graph.edge_index:
        temp_edge_index = het_graph.edge_index[key]
        from_type = key[0]
        to_type = key[2]
        adj = SparseTensor(row=temp_edge_index[0], col=temp_edge_index[1], sparse_sizes=(het_graph.num_nodes(from_type), het_graph.num_nodes(to_type)))
        sparse_edge_dict[key] = adj.t()
        het_graph["sparse_mat"] = sparse_edge_dict

#hete.apply_transform(edgeindex_to_sparsematrix,update_graph=True, update_tensor=False)

task = 'link_pred'
dataset = GraphDataset([hete], task=task, edge_train_mode="all")
dataset_train, dataset_val, dataset_test = dataset.split(transductive=True, split_ratio=[0.4, 0.3, 0.3])

# dataset_train.apply_transform(edgeindex_to_sparsematrix, deep_copy=True)
# dataset_val.apply_transform(edgeindex_to_sparsematrix)
# dataset_test.apply_transform(edgeindex_to_sparsematrix)
# hete.apply_transform(edgeindex_to_sparsematrix)


dataset_train.apply_transform(edgeindex_to_sparsematrix)
dataset_val.apply_transform(edgeindex_to_sparsematrix)
dataset_test.apply_transform(edgeindex_to_sparsematrix)
hete.apply_transform(edgeindex_to_sparsematrix)
#para ver como quedo
#dataset_train.graphs[0].keys
dataset_dict = {"train":dataset_train,"val":dataset_val,"test":dataset_test}

#%%
#region train
args = {
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'hidden_size': 32,
    'epochs': 100,
    'weight_decay': 1e-5,
    'lr': 0.003,
}
model = HeteroGNN(hete, args, aggr="mean")
optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

#%%
loss_plot = []
epochs = 2000
for epoch in range(epochs):
    if epoch%10 == 0:
       loss = train(model,optimizer,dataset_dict["train"],printb=True)
    else:
        loss = train(model,optimizer,dataset_dict["train"],printb=False)
    loss_plot.append(loss)

plt.plot(np.arange(len(loss_plot)),loss_plot)
#%%
#veo una predicción
p = model(dataset_val[0])
prediction = {edgetype:torch.round(torch.sigmoid(pred)) for edgetype,pred in p.items()}
ground_truth = dataset_val[0].edge_label

test(model,dataset_dict,args)

#%%