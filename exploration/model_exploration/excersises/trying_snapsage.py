#%%
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
#import deepsnap
import torch.nn as nn
#import torch.nn.functional as F
#from sklearn.metrics import f1_score

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

from torch.utils.tensorboard import SummaryWriter
#%%
def generate_2convs_link_pred_layers(hete, conv, hidden_size):
    convs1 = {}
    convs2 = {}
    for message_type in hete.message_types:
        n_type = message_type[0]
        s_type = message_type[2]
        n_feat_dim = hete.num_node_features(n_type)
        s_feat_dim = hete.num_node_features(s_type)
        convs1[message_type] = conv(n_feat_dim, hidden_size, s_feat_dim)
        convs2[message_type] = conv(hidden_size, hidden_size, hidden_size)
    return convs1, convs2

class HeteroGNN2(torch.nn.Module):
    def __init__(self, conv1, conv2, hetero, hidden_size):
        super(HeteroGNN2, self).__init__()
        
        self.convs1 = HeteroConv(conv1) # Wrap the heterogeneous GNN layers
        self.convs2 = HeteroConv(conv2)
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.bns1 = nn.ModuleDict()
        self.bns2 = nn.ModuleDict()
        self.relus1 = nn.ModuleDict()
        self.relus2 = nn.ModuleDict()
        self.post_mps = nn.ModuleDict()

        for node_type in hetero.node_types:
            self.bns1[node_type] = torch.nn.BatchNorm1d(hidden_size)
            self.bns2[node_type] = torch.nn.BatchNorm1d(hidden_size)
            self.relus1[node_type] = nn.LeakyReLU()
            self.relus2[node_type] = nn.LeakyReLU()

    def forward(self, data):
        x = data.node_feature
        edge_index = data.edge_index
        x = self.convs1(x, edge_index)
        x = forward_op(x, self.bns1)
        x = forward_op(x, self.relus1)
        x = self.convs2(x, edge_index)
        x = forward_op(x, self.bns2)

        pred = {}
        for message_type in data.edge_label_index:
            nodes_first = torch.index_select(x['n1'], 0, data.edge_label_index[message_type][0,:].long())
            nodes_second = torch.index_select(x['n1'], 0, data.edge_label_index[message_type][1,:].long())
            pred[message_type] = torch.sum(nodes_first * nodes_second, dim=-1)
        return pred

    def loss(self, pred, y):
        loss = 0
        for key in pred:
            p = torch.sigmoid(pred[key])
            loss += self.loss_fn(p, y[key].type(pred[key].dtype))
        return loss
#%%
hidden_size = 32
hetero = hete
conv1, conv2 = generate_2convs_link_pred_layers(hetero, HeteroSAGEConv, hidden_size)
model = HeteroGNN2(conv1, conv2, hetero, hidden_size)
optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
#%%
traindata = DataLoader(dataset_train, collate_fn=Batch.collate(), batch_size=1)
for i,data in enumerate(traindata):
    model(data)