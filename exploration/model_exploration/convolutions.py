import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_sparse import matmul
import torch_geometric.nn as pyg_nn
import deepsnap.hetero_gnn
from prediction_heads import distmult_head
from utils import edgeindex_to_sparsematrix

quiet = False

def talk(msg, quiet=quiet):
    if not quiet:
        print(msg)

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


class HeteroGNNConv(pyg_nn.MessagePassing):
    def __init__(self, in_channels_src, in_channels_dst, out_channels):
        super().__init__(aggr="mean")

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

        talk("HeteroGNN forward")
        talk(f"Node feature src shape: {node_feature_src.shape}, Node feature dst shape: {node_feature_dst.shape}, edge index shape: {edge_index.sparse_sizes()}")
        out = self.propagate(edge_index, size, node_feature_src=node_feature_src, node_feature_dst=node_feature_dst)
        return out

    def message_and_aggregate(self, edge_index, node_feature_src):
        talk("HeteroGNN msg and agg")
        out = matmul(edge_index, node_feature_src, reduce=self.aggr)
        return out

    def update(self, aggr_out, node_feature_dst):
        talk("HeteroGNN update")
        talk(f"Aggr_out shape: {aggr_out.shape}")
        talk(f"Dst feature shape: {node_feature_dst.shape}")
        dst_msg = self.lin_dst(node_feature_dst)
        src_msg = self.lin_src(aggr_out)

        talk(f"Concat: dst_msg shape: {dst_msg.shape}, src_msg shape: {src_msg.shape}")
        full_msg = torch.concat((dst_msg, src_msg), dim=1)
        out = self.lin_update(full_msg)
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
        talk("Wrapper forward")
        message_type_emb = {}

        for message_key, adj in edge_indices.items():
            talk(f"\n{message_key}\n")
            src_type, edge_type, dst_type = message_key
            node_feature_src = node_features[src_type]
            node_feature_dst = node_features[dst_type]
            message_type_emb[message_key] = (self.convs[message_key](node_feature_src,node_feature_dst,adj))

        node_emb = {dst: [] for _, _, dst in message_type_emb.keys()}
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
        return node_emb

    def aggregate(self, xs):
        talk("Wrapper agg") 
        return torch.mean(torch.stack(xs, dim=-1), dim=-1)



class HeteroGNN(torch.nn.Module):
    def __init__(self, hetero_graph, head, pred_mode, args, aggr="mean"):
        super().__init__()

        self.aggr = aggr
        self.head = head
        self.pred_mode = pred_mode
        self.hidden_size = args['hidden_size']
        self.bns1 = torch.nn.ModuleDict()
        self.bns2 = torch.nn.ModuleDict()
        self.relus1 = torch.nn.ModuleDict()
        self.relus2 = torch.nn.ModuleDict()
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        
        if head=="dismult":
          self.distmult_head = distmult_head(hetero_graph,self.hidden_size)

        convs1 = generate_convs(hetero_graph, HeteroGNNConv, self.hidden_size, first_layer=True)
        convs2 = generate_convs(hetero_graph, HeteroGNNConv, self.hidden_size, first_layer=False)
        self.convs1 = HeteroGNNWrapperConv(convs1, args, aggr=self.aggr)
        self.convs2 = HeteroGNNWrapperConv(convs2, args, aggr=self.aggr)
        for node_type in hetero_graph.node_types:
            self.bns1[node_type] = torch.nn.BatchNorm1d(self.hidden_size)
            self.bns2[node_type] = torch.nn.BatchNorm1d(self.hidden_size)
            self.relus1[node_type] = torch.nn.LeakyReLU()
            self.relus2[node_type] = torch.nn.LeakyReLU()
    

    def forward(self, graph):
        talk("General forward")
        x, edge_label_index = graph.node_feature, graph.edge_label_index
        adj = edgeindex_to_sparsematrix(graph)
        talk("Conv 1")
        x = self.convs1(x, edge_indices=adj)
        talk("BNS 1")
        x = deepsnap.hetero_gnn.forward_op(x, self.bns1)
        talk("Relu 1")
        x = deepsnap.hetero_gnn.forward_op(x, self.relus1)
        talk("Conv 2")
        x = self.convs2(x, edge_indices=adj)
        talk("BNS 2")
        x = deepsnap.hetero_gnn.forward_op(x, self.bns2)


        if self.head == "dotprod":
          talk("General decoder")
          pred = {}
          if self.pred_mode == "all":
            for message_type in edge_label_index:
                src_type = message_type[0]
                trg_type = message_type[2]
                nodes_first = torch.index_select(x[src_type], 0, edge_label_index[message_type][0,:].long())
                nodes_second = torch.index_select(x[trg_type], 0, edge_label_index[message_type][1,:].long())
                pred[message_type] = torch.sum(nodes_first * nodes_second, dim=-1)
          elif self.pred_mode == "gda_only":
            keys = [('gene_protein', 'gda', 'disease'), ('disease', 'gda', 'gene_protein')]
            for message_type in keys:
              src_type = message_type[0]
              trg_type = message_type[2]
              nodes_first = torch.index_select(x[src_type], 0, edge_label_index[message_type][0,:].long())
              nodes_second = torch.index_select(x[trg_type], 0, edge_label_index[message_type][1,:].long())
              pred[message_type] = torch.sum(nodes_first * nodes_second, dim=-1)
          return pred
        elif self.head == "distmult":
          return self.distmult_head.score(x,edge_label_index)
          
    def loss(self, pred, y):
        loss = 0
        sets = torch.tensor(len(pred.keys()))
        for key in pred:
            p = pred[key]
            loss += self.loss_fn(p, y[key].type(pred[key].dtype))
        return loss
