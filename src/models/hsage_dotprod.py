# Implementación de un hetero graphSAGE o Relational GraphSAGE + un decoder distmult al estilo Schlichtkrull et. al
# Sin mecanismos de atención
# Mean aggregation only
# %%
import torch
import copy
import torch
import deepsnap
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from sklearn.metrics import f1_score
from deepsnap.hetero_gnn import forward_op
from torch_sparse import matmul
# %%
debugging = False


def my_debug(arg):
    if debugging:
        print(arg)
# %%
# region MODEL CLASS DEFINITION

class HeteroGNNConv(pyg_nn.MessagePassing):
    def __init__(self, in_channels_src, in_channels_dst, out_channels):
        super(HeteroGNNConv, self).__init__(aggr="mean")

        self.in_channels_src = in_channels_src
        self.in_channels_dst = in_channels_dst
        self.out_channels = out_channels
        self.lin_dst = nn.Linear(in_channels_dst, out_channels)
        self.lin_src = nn.Linear(in_channels_src, out_channels)
        self.lin_update = nn.Linear(2*out_channels, out_channels)

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
        my_debug(f'message_and_agregate out: {out}')

        return out

    def update(self, aggr_out, node_feature_dst):

        dst_msg = self.lin_dst(node_feature_dst)
        src_msg = self.lin_src(aggr_out)
        full_msg = torch.concat((dst_msg, src_msg), dim=1)
        out = self.lin_update(full_msg)
        my_debug(f'update out: {out}')
        return out


class HeteroGNNWrapperConv(deepsnap.hetero_gnn.HeteroConv):
    def __init__(self, convs, args, aggr="mean"):
        super(HeteroGNNWrapperConv, self).__init__(convs, None)
        self.aggr = aggr

        # Map the index and message type
        self.mapping = {}

    def reset_parameters(self):
        super(HeteroGNNWrapperConv, self).reset_parameters()

    def forward(self, node_features, edge_indices):
        message_type_emb = {}
        my_debug(f'Wrapper self convs: {self.convs}')
        for message_key, message_type in edge_indices.items():
            src_type, edge_type, dst_type = message_key
            # my_debug(message_key)
            # my_debug(node_features.shape)
            my_debug(f'Wrapper self conv step: {self.convs[message_key]}')
            node_feature_src = node_features[src_type]
            node_feature_dst = node_features[dst_type]
            edge_index = edge_indices[message_key]
            my_debug(
                f'Wrapper message_type_emb step: {self.convs[message_key](node_feature_src,node_feature_dst,edge_index)}')
            message_type_emb[message_key] = (
                self.convs[message_key](
                    node_feature_src,
                    node_feature_dst,
                    edge_index,
                )
            )
            my_debug(f'{message_key} emb done: {message_type_emb}')
        node_emb = {dst: [] for _, _, dst in message_type_emb.keys()}
        my_debug(f'node emb: {node_emb}')
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
        my_debug(f'node_emb from wrapper conv forward:{node_emb}')
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
        super(HeteroGNN, self).__init__()

        self.aggr = aggr
        self.hidden_size = args['hidden_size']
        self.bns1 = nn.ModuleDict()
        self.bns2 = nn.ModuleDict()
        self.relus1 = nn.ModuleDict()
        self.relus2 = nn.ModuleDict()
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        #self.post_mps = nn.ModuleDict()

        convs1 = generate_convs(hetero_graph, HeteroGNNConv, self.hidden_size, first_layer=True)
        my_debug(f'convs1: {convs1}')
        convs2 = generate_convs(hetero_graph, HeteroGNNConv, self.hidden_size, first_layer=False)
        self.convs1 = HeteroGNNWrapperConv(convs1, args, aggr=self.aggr)
        my_debug(f'self.convs1: {self.convs1}')
        self.convs2 = HeteroGNNWrapperConv(convs2, args, aggr=self.aggr)
        for node_type in hetero_graph.node_types:
            self.bns1[node_type] = torch.nn.BatchNorm1d(self.hidden_size, eps=1)
            self.bns2[node_type] = torch.nn.BatchNorm1d(self.hidden_size, eps=1)
            self.relus1[node_type] = nn.LeakyReLU()
            self.relus2[node_type] = nn.LeakyReLU()

    def forward(self,hetero_graph):
        x, edge_index, edge_label_index = hetero_graph.node_feature, hetero_graph.edge_index, hetero_graph.edge_label_index

        x = self.convs1(x, edge_indices=edge_index)
        x = forward_op(x, self.bns1)
        x = forward_op(x, self.relus1)
        x = self.convs2(x, edge_indices=edge_index)
        x = forward_op(x, self.bns2)
        x = forward_op(x, self.relus2)
        #x = forward_op(x, self.post_mps)

        pred = {}
        for message_type in edge_label_index:
            nodes_first = torch.index_select(x['n1'], 0, edge_label_index[message_type][0,:].long())
            nodes_second = torch.index_select(x['n1'], 0, edge_label_index[message_type][1,:].long())
            pred[message_type] = torch.sum(nodes_first * nodes_second, dim=-1)
        return pred

    def loss(self, pred, y):
        loss = 0
        for key in pred:
            p = torch.sigmoid(pred[key])
            loss += self.loss_fn(p, y[key].type(pred[key].dtype))
        return loss
# endregion
# %%
# region DEF TRAIN AND TEST FUNCTIONS


def train(model, optimizer, hetero_graph):
    model.train()
    optimizer.zero_grad()
    preds = model(hetero_graph)
    loss = model.loss(preds, hetero_graph.edge_label)
    loss.backward()
    optimizer.step()
    return loss.item()


def test(model, graph, indices, best_model=None, best_val=0, save_preds=False, agg_type=None):
    model.eval()
    accs = []
    for i, index in enumerate(indices):
        preds = model(graph.node_feature, graph.edge_index)
        num_node_types = 0
        micro = 0
        macro = 0
        for node_type in preds:
            idx = index[node_type]
            pred = preds[node_type][idx]
            pred = pred.max(1)[1]
            label_np = graph.node_label[node_type][idx].cpu().numpy()
            pred_np = pred.cpu().numpy()
            micro = f1_score(label_np, pred_np, average='micro')
            macro = f1_score(label_np, pred_np, average='macro')
            num_node_types += 1

        # Averaging f1 score might not make sense, but in our example we only
        # have one node type
        micro /= num_node_types
        macro /= num_node_types
        accs.append((micro, macro))

        # Only save the test set predictions and labels!
        if save_preds and i == 2:
            print(
                "Saving Heterogeneous Node Prediction Model Predictions with Agg:", agg_type)
            print()

            data = {}
            data['pred'] = pred_np
            data['label'] = label_np

            df = pd.DataFrame(data=data)
            # Save locally as csv
            df.to_csv('ACM-Node-' + agg_type + 'Agg.csv', sep=',', index=False)

    if accs[1][0] > best_val:
        best_val = accs[1][0]
        best_model = copy.deepcopy(model)
    return accs, best_model, best_val
# endregion

# %%
