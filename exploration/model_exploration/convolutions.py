#%%
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

def hetero_apply_function(x: dict,func) -> dict:
    """X es el diccionario de node embeddings o features, {node_type: tensor}.
    Aplica func a cada entrada del diccionario, devuelve un dict de la misma forma."""
    x_transformed = {}
    for key,val in x.items():
        transformed_val = func(val)
        x_transformed[key] = transformed_val
    
    return x_transformed

class HeteroGNNConv(pyg_nn.MessagePassing):
    def __init__(self, in_channels_src, in_channels_dst, out_channels):
        super().__init__(aggr="mean")

        self.in_channels_src = in_channels_src
        self.in_channels_dst = in_channels_dst
        self.out_channels = out_channels

        self.lin_dst = nn.Linear(in_channels_dst, out_channels)
        self.lin_src = nn.Linear(in_channels_src, out_channels)
        self.lin_update = nn.Linear(2*out_channels, out_channels)

    def forward(self,node_feature_src, node_feature_dst,edge_index):
        talk("HeteroGNN forward")
        talk(f"Node feature src shape: {node_feature_src.shape}, Node feature dst shape: {node_feature_dst.shape}, edge index shape: {edge_index.sparse_sizes()}")
        out = self.propagate(edge_index, node_feature_src=node_feature_src,node_feature_dst=node_feature_dst)
        return out

    def message_and_aggregate(self, edge_index, node_feature_src):
        talk("HeteroGNN msg and agg")
        talk(f"node_feature src shape: {node_feature_src.shape}")
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

        talk(f"Full msg shape: {full_msg.shape}")
        out = self.lin_update(full_msg)

        talk(f"After update shape: {out.shape}")
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
        talk("\n ------ Wrapper forward ------ ")
        message_type_emb = {}

        for message_key, adj in edge_indices.items():
            talk(f"\n{message_key}\n")
            src_type, edge_type, dst_type = message_key
            node_feature_src = node_features[src_type]
            node_feature_dst = node_features[dst_type]

            message_type_emb[message_key] = self.convs[message_key](node_feature_src,node_feature_dst,adj)

        # {dst: [] for src, type, dst in message_type.emb.keys()}
        # {tipo de nodo: [lista de embeddings obtenidos]}
        node_emb = {dst: [] for _, _, dst in message_type_emb.keys()}
        mapping = {}

        for (src, edge_type, dst), item in message_type_emb.items():
            #esto es para saber que indice es cada terna/msg type
            mapping[len(node_emb[dst])] = (src, edge_type, dst)

            #Agrego el embedding de la terna (src,type,dst) al la lista de embeddings de dst
            node_emb[dst].append(item)

        self.mapping = mapping

        #Ahora hago aggregation sobre las listas de embeddings, para cada tipo de nodo DST
        talk("\n------ Wrapper agg ------")
        for node_type, embs in node_emb.items():
            talk(f"\nAggregate {node_type} embeddings")

            # Si hay un solo embedding en la lista, me quedo con ese solito
            if len(embs) == 1:
                talk(f"Num embeddings = 1, no AGG needed")
                node_emb[node_type] = embs[0]
            
            #Si hay más de uno hago aggregation
            else:
                node_emb[node_type] = self.aggregate(embs)
        return node_emb

    def aggregate(self, xs):
        # Tomo la lista de embeddings para cada tipo de nodo y los "agrego". En este caso solo los promedio
        # Stackeo los embeddings
        talk(f"Num embeddings: {len(xs)}")
        talk(f"Shape embeddings: {[e.shape for e in xs]}")
        stacked = torch.stack(xs, dim=0)
        talk(f"Stacked shape: {stacked.shape}")
        out = torch.mean(stacked,dim=0)
        talk(f"Final aggregated shape: {out.shape}")
        return out



class HeteroGNN(torch.nn.Module):
    def __init__(self, hetero_graph, head, pred_mode, args, aggr="mean"):
        super().__init__()

        self.aggr = aggr
        self.head = head
        self.pred_mode = pred_mode
        self.hidden_size = args['hidden_size']
        self.bns1 = torch.nn.ModuleDict()
        self.relus1 = torch.nn.ModuleDict()
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        
        if head=="distmult":
          self.distmult_head = distmult_head(hetero_graph,self.hidden_size)

        convs1 = generate_convs(hetero_graph, HeteroGNNConv, self.hidden_size, first_layer=True)
        convs2 = generate_convs(hetero_graph, HeteroGNNConv, self.hidden_size, first_layer=False)
        self.convs1 = HeteroGNNWrapperConv(convs1, args, aggr=self.aggr)
        self.convs2 = HeteroGNNWrapperConv(convs2, args, aggr=self.aggr)
        for node_type in hetero_graph.node_types:
            self.bns1[node_type] = torch.nn.BatchNorm1d(self.hidden_size)
            self.relus1[node_type] = torch.nn.LeakyReLU()
    

    def forward(self, graph):
        talk(" ------ ENCODER ------ ")
        x, edge_label_index = graph.node_feature, graph.edge_label_index
        adj = edgeindex_to_sparsematrix(graph)
        talk("Conv 1")
        x = self.convs1(x, edge_indices=adj)

        talk("\n BNS 1")
        x = deepsnap.hetero_gnn.forward_op(x, self.bns1)

        talk("\n Relu 1")
        x = deepsnap.hetero_gnn.forward_op(x, self.relus1)

        talk("\n Conv 2")
        x = self.convs2(x, edge_indices=adj)

        talk("\n----------")
        talk(f"Node embeddings done. Dimentions: {[(k,item.shape) for k,item in x.items()]}")
        talk("---------")
        # talk("Normalizing embeddings")
        # x = hetero_apply_function(x,torch.nn.functional.normalize)


        if self.head == "dotprod":
          talk("\n ------ DECODER ------ ")
          pred = {}
          if self.pred_mode == "all":
            for message_type, edge_index in edge_label_index.items():
                talk(f"\n Decoding edge type: {message_type}")
                src_type = message_type[0]
                trg_type = message_type[2]

                x_source = x[src_type]
                x_target = x[trg_type]

                nodes_src = x_source[edge_index[0]]
                nodes_trg = x_target[edge_index[1]]

                talk(f"\n Multiplying shapes: {nodes_src.shape}, {nodes_trg.shape}")
                pred[message_type] = torch.sum(nodes_src * nodes_trg, dim=-1)

          elif self.pred_mode == "gda_only":
            keys = [('gene_protein', 'gda', 'disease'), ('disease', 'gda', 'gene_protein')]
            for message_type in keys:
                talk(f"\n Decoding edge type: {message_type}")
                edge_index = edge_label_index[message_type]
                src_type = message_type[0]
                trg_type = message_type[2]

                x_source = x[src_type]
                x_target = x[trg_type]

                nodes_src = x_source[edge_index[0]]
                nodes_trg = x_target[edge_index[1]]
                talk(f"\n Multiplying shapes: {nodes_src.shape}, {nodes_trg.shape}")
                pred[message_type] = torch.sum(nodes_src * nodes_trg, dim=-1)

          return pred

        elif self.head == "distmult":
          return self.distmult_head.score(x,edge_label_index)
          
    def loss(self, prediction_dict, ground_truth_dict):
        loss = 0
        num_types = len(prediction_dict.keys())
        # sets = torch.tensor(len(pred.keys()))
        for edge_type,pred in prediction_dict.items():
            y = ground_truth_dict[edge_type]
            loss += self.loss_fn(pred, y.type(pred.dtype))
        return loss/num_types