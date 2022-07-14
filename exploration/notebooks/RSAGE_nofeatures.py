# %%
#Librería base de redes neuronales de torch
#Acá están las clases "base" de las que heredan todos los modelos
import torch
import torch.nn as nn

#Operaciones entre sparse matrix
from torch_sparse import matmul, SparseTensor

#Librerías específicas de GNNs: 
#PYG es la libraría general de GNNs,
#DeepSNAP tiene utilidades para el manejo de datos y grafos heterogeneos, hacer splits, negative sampling, etc.
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import HeteroConv
from deepsnap.dataset import GraphDataset
import deepsnap.hetero_gnn
from deepsnap.hetero_gnn import forward_op
from deepsnap.hetero_graph import HeteroGraph
from torch_geometric.utils import to_undirected
import torch.nn.functional as F
#from torch_geometric.data import HeteroData

#Para manejar los minibatches
from torch.utils.data import DataLoader
from deepsnap.batch import Batch

from timeit import default_timer as timer
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import copy

#region def util functions
#%%
def get_prediction(model,H:HeteroGraph) -> dict:
    prediction = {}
    with torch.no_grad():
        preds = model(H)
        for key,pred in preds.items():
            logits = torch.sigmoid(pred)
            pred_label = torch.round(logits)
            prediction[key] = pred_label
    return prediction

def get_edge_data(H:HeteroGraph, type:tuple) -> dict:
    """(n1,n2): {attr dict}"""
    edges = list(H.G.edges(data=True))
    edge_mapping = H.edge_to_graph_mapping[type].tolist()
    #data = {(edges[idx][0], edges[idx][1]):edges[idx][2] for j,idx in edge_mapping}
    data = {j:{"graph_idx":idx, "edge": (edges[idx][0],edges[idx][1]), "attr":edges[idx][2]} for j,idx in enumerate(edge_mapping)}
    return data

def get_node_data(H:HeteroGraph, type:str) -> dict:
    """{tensor idx: {attribute dict})}"""
    nodes = dict(H.G.nodes(data=True))
    node_mapping = H.node_to_graph_mapping[type].tolist()
    data = {j:{"data":nodes[idx], "graph_idx":idx} for j,idx in enumerate(node_mapping)}
    return data

def check_same_edges(H:HeteroGraph, type1:tuple, type2:tuple) -> bool:
    edges1 = list(get_edge_data(H,type1).keys())
    edges2 = list(get_edge_data(H,type2).keys())
    reversed_1 = [tuple(reversed(edge)) for edge in edges1]
    return set(reversed_1) == set(edges2)

def tensor_to_edgelist(tensor: torch.tensor):
  "Toma un edge_index de shape (2,num_edges) y devuelve una lista de tuplas"
  sources = tensor[0,:].tolist()
  targets = tensor[1,:].tolist()
  edgelist = list(zip(sources,targets))
  return edgelist

def init_node_features(G, mode, size):
  """ Para usar con nx.Graph """
  if mode == "ones":
    feature = torch.ones(size)
    nx.set_node_attributes(G, feature, 'node_feature')
  elif mode == "random":
    feature_dict = {}
    for node in list(G.nodes()):
      feature_dict[node] = torch.rand(size)
    nx.set_node_attributes(G,feature_dict,'node_feature')

def map_prediction_to_edges(prediction:torch.tensor,H:HeteroGraph, edge_to_label = True) -> dict:
  """Dada una predicción, devuelve un diccionario que mapea el edge (u,v) a la etiqueta predecida.
  El mapa corresponde a los enlaces en el dataset usado: si se predijo sobre val, pasar dataset val al argument
  esto es importante porque los indices no se conservan en los splits
  if edge_to_label el dict es {(u,v):label}, sino es {label:[(u,v),...]}"""
  prediction_map = {}
  for edge_type,pred in prediction.items():
    predicted_labels = pred.tolist()
    labeled_edges = tensor_to_edgelist(H.edge_label_index[edge_type])
    pred_map = {edge:label for edge,label in zip(labeled_edges,predicted_labels)}
    if edge_to_label:
      prediction_map[edge_type] = pred_map
    else:
      positives = [edge for edge,val in pred_map.items() if val==1]
      negatives = [edge for edge,val in pred_map.items() if val==0]
      prediction_map[edge_type] = {1:positives, 0:negatives}
  return prediction_map

def get_edge(data:pd.DataFrame,edge:tuple):
    row = edge_data[(data.a_idx == edge[0]) & (data.b_idx == edge[1])]
    if row.empty:
      row = edge_data[(data.a_idx == edge[1]) & (data.b_idx == edge[0])]
      if row.empty:
        print("Edge does not exist in dataframe")
    return row

def get_prediction_data_dict(prediction,dataset,nodes_data)-> dict:
  """input indexado con tensor indexes"""
  results = {}
  edge_to_label_dict = map_prediction_to_edges(prediction,dataset)
  for edgetype,pred in edge_to_label_dict.items():
    src_type = edgetype[0]
    trg_type = edgetype[2]
    src_info = nodes_data[src_type]
    trg_info = nodes_data[trg_type]
    edgetype_results = {edge:{"source_data":src_info[edge[0]], "target_data":trg_info[edge[1]], "label":label} for edge,label in pred.items()}
    results[edgetype] = edgetype_results
  return results

def get_prediction_dataframe(prediction,dataset)-> dict:
  """input indexado con tensor indexes"""
  #TODO: agregar una columna de score
  results = {}
  edge_to_label_dict = map_prediction_to_edges(prediction,dataset)
  for edgetype,pred in edge_to_label_dict.items():
    src_type, relation, trg_type = edgetype
    src_info = get_node_data(dataset,src_type)
    trg_info = get_node_data(dataset,trg_type)
    for edge, label in pred.items():
      src,trg = edge
      results[edge] = {"type":relation,"source_idx":src_info[src]["data"]["node_dataset_idx"],"target_idx":trg_info[trg]["data"]["node_dataset_idx"],"source_type":src_info[src]["data"]["node_type"],"target_type":trg_info[trg]["data"]["node_type"],"source_name":src_info[src]["data"]["node_name"], "target_name":trg_info[trg]["data"]["node_name"] ,"label":label}
    frame = pd.DataFrame.from_dict(results, orient="index")
  return frame
#endregion
#region Modelo
# %%
class distmult_head(torch.nn.Module):
  def __init__(self, hetero_graph, hidden_size):
    super().__init__()
    self.R_weights = nn.ParameterDict()

    for edge_type in hetero_graph.edge_types:
      self.R_weights[edge_type] = nn.Parameter(torch.rand(hidden_size,hidden_size)*0.01)
  
  def score(self,x,edge_label_index):
    scores = {}
    for message_type in edge_label_index:
      src_type,edge_type,trg_type = message_type[0], message_type[1], message_type[2]
      rel_weights = self.R_weights[edge_type]
      nodes_left = torch.index_select(x[src_type], 0, edge_label_index[message_type][0,:].long())
      nodes_right = torch.index_select(x[trg_type], 0, edge_label_index[message_type][1,:].long())
      mid_product = nodes_right@rel_weights
      scores[message_type] = torch.sum(mid_product@torch.t(nodes_left) , dim=-1)
    return scores

# %%
def edgeindex_to_sparsematrix(het_graph: HeteroGraph) -> dict : 
    sparse_edge_dict = {}
    for key in het_graph.edge_index:
        temp_edge_index = het_graph.edge_index[key]
        from_type = key[0]
        to_type = key[2]
        adj = SparseTensor(row=temp_edge_index[0], col=temp_edge_index[1], sparse_sizes=(het_graph.num_nodes(from_type), het_graph.num_nodes(to_type)))
        sparse_edge_dict[key] = adj.t()
    return sparse_edge_dict

debugging = False

def my_debug(arg):
    if debugging:
        print(arg)


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

        out = self.propagate(edge_index, size, node_feature_src=node_feature_src, node_feature_dst=node_feature_dst)
        return out

    def message_and_aggregate(self, edge_index, node_feature_src):
        out = matmul(edge_index, node_feature_src, reduce=self.aggr)
        return out

    def update(self, aggr_out, node_feature_dst):
        dst_msg = self.lin_dst(node_feature_dst)
        src_msg = self.lin_src(aggr_out)
        full_msg = torch.concat((dst_msg, src_msg), dim=-1)
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
        message_type_emb = {}

        for message_key, message_type in edge_indices.items():
            src_type, edge_type, dst_type = message_key
            node_feature_src = node_features[src_type]
            node_feature_dst = node_features[dst_type]
            edge_index = edge_indices[message_key]
            message_type_emb[message_key] = (self.convs[message_key](node_feature_src,node_feature_dst,edge_index))

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
        return torch.mean(torch.stack(xs, dim=-1), dim=-1)

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
        x, edge_label_index = graph.node_feature, graph.edge_label_index
        adj = edgeindex_to_sparsematrix(graph)
        x = self.convs1(x, edge_indices=adj)
        x = deepsnap.hetero_gnn.forward_op(x, self.bns1)
        x = deepsnap.hetero_gnn.forward_op(x, self.relus1)
        x = self.convs2(x, edge_indices=adj)
        x = deepsnap.hetero_gnn.forward_op(x, self.bns2)


        if self.head == "dotprod":
          pred = {}
          if self.pred_mode == "all":
            for message_type in edge_label_index:
                src_type = message_type[0]
                trg_type = message_type[2]
                nodes_first = torch.index_select(x[src_type], 0, edge_label_index[message_type][0,:].long())
                nodes_second = torch.index_select(x[trg_type], 0, edge_label_index[message_type][1,:].long())
                pred[message_type] = torch.sum(nodes_first * nodes_second, dim=-1)
          elif self.pred_mode == "gda_only":
            keys = [('gene/protein', 'GDA', 'disease'), ('disease', 'GDA', 'gene/protein')]
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

#endregion
#region Preprocesamiento del dataset
#%%
node_data = pd.read_csv("../../data/processed/graph_node_table.csv", index_col=0)
edge_data = pd.read_csv("../../data/processed/graph_edge_table.csv",index_col=0).rename(columns={"relation":"edge_type"})
#%%
D = nx.from_pandas_edgelist(edge_data,source="a_idx",target="b_idx", edge_attr="edge_type")
nx.set_node_attributes(D,pd.Series(node_data.node_type, index=node_data.node_idx).to_dict(),"node_type")
nx.set_node_attributes(D,pd.Series(node_data.node_name, index=node_data.node_idx).to_dict(),"node_name")
nx.set_node_attributes(D,pd.Series(node_data.node_idx, index=node_data.node_idx).to_dict(),"node_dataset_idx")
#nx.set_edge_attributes(D,pd.Series(edge_data.edge_idx, index=edge_data.edge_idx).to_list(), "edge_dataset_idx")

G = D.to_directed()

#endregion
#region train and eval func
# %%
def train(model, optimizer, graph, printb):
    model.train()
    optimizer.zero_grad()
    preds = model(graph)
    loss = model.loss(preds, graph.edge_label)
    loss.backward()
    optimizer.step()
    acc = 0
    num = 0
    for key,pred in preds.items():
      logits = torch.sigmoid(pred)
      pred_label = torch.round(logits)
      acc += (pred_label == graph.edge_label[key]).sum().item()
      num += pred_label.shape[0]
    accuracy = acc/num
    if printb:
        print(loss.item())
    return loss.item(), accuracy


# Test function
def test(model, splits_dict, args):
    model.eval()
    accs = {}
    for mode, dataset in splits_dict.items():
        acc = 0
        num = 0
        pred = model(dataset)
        for key in pred:
            p = torch.sigmoid(pred[key]).cpu().detach()
            pred_label = np.zeros_like(p, dtype=np.int64)
            pred_label[np.where(p > 0.5)[0]] = 1
            pred_label[np.where(p <= 0.5)[0]] = 0
            acc += np.sum(pred_label == dataset.edge_label[key].cpu().numpy())
            num += len(pred_label)
        accs[mode] = acc / num
    return accs

def test2(model,validation_set,global_accuracy=True):
  model.eval()
  if global_accuracy:
    acc = 0
    num = 0
    with torch.no_grad():
      preds = model(validation_set)
      for key,pred in preds.items():
        logits = torch.sigmoid(pred)
        pred_label = torch.round(logits)
        acc += (pred_label == validation_set.edge_label[key]).sum().item()
        num += pred_label.shape[0]
    accuracy = round(acc/num,3)
    return accuracy
  else:
    type_accuracy = {}
    with torch.no_grad():
      preds = model(validation_set)
      for key,pred in preds.items():
        logits = torch.sigmoid(pred)
        pred_label = torch.round(logits)
        acc = (pred_label == validation_set.edge_label[key]).sum().item()
        num = pred_label.shape[0]
        type_accuracy[key] = round(acc/num,3)
    return type_accuracy

def hits_at_k(model,dataset,k,return_indices=False) -> dict:
  hits = {}
  with torch.no_grad():
    preds = model(dataset)
    for key,pred in preds.items():
        #ordeno los puntajes de mayor a menor
        pred, indices = torch.sort(pred, descending=True)

        #corto el ranking en 0.5
        pred = pred[pred>0.5]

        #me quedo solo con los k mayor punteados
        if pred.shape[0]>k:
          pred, indices = pred[:k].to(args["device"]), indices[:k].to(args["device"])
        else:
          print(f"Top {k} scores below classification threshold 0.5, returning top {pred.shape[0]}")

        #busco que label tenían esas k preds
        labels = torch.index_select(dataset.edge_label[key],-1,indices)

        #cuento cuantas veces predije uno positivo en el top k
        if return_indices:
          #devuelvo los indices por si quiero ir a buscar los enlaces al grafo
          hits[key] = {"hits":labels.sum().item(),"indices":indices}
        else:
          #si solo quiero contar hits no devuelvo indices
          hits[key] = {"hits":labels.sum().item()}

  return hits
#endregion
#region split dataset
# %%
args = {
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'hidden_size': 32,
    'weight_decay': 1e-5,
    'lr': 0.001,
}
task = 'link_pred'
train_mode = "disjoint"

dataset = GraphDataset([Hete], task=task, edge_train_mode=train_mode, resample_negatives=True)
dataset_train, dataset_val, dataset_test = dataset.split(transductive=True, split_ratio=[0.4, 0.3, 0.3])
splits = {"train":dataset_train[0].to(args["device"]), "val":dataset_val[0].to(args["device"]), "test":dataset_test[0].to(args["device"])}
#endregion
#region init model
# %%
model = HeteroGNN(Hete,"dotprod", "all", args, aggr="mean",).to(args["device"])
optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
#endregion
#region train model
# %%
loss_plot = []
train_acc = []
val_acc = []
epochs = 100
for epoch in range(epochs):
    if epoch%10 == 0:
       loss,t_acc = train(model,optimizer,splits["train"],printb=True)
       v_acc = test2(model,splits["val"])
    else:
        loss,t_acc = train(model,optimizer,splits["train"],printb=False)
        v_acc = test2(model,splits["val"])
    loss_plot.append(loss)
    train_acc.append(t_acc)
    val_acc.append(v_acc)

fig,axs = plt.subplots(1, 2, figsize=(18,6))
axs[0].plot(np.arange(epochs),loss_plot)
axs[0].set_xlabel("Epoch")
axs[0].set_ylabel("Training Loss")
axs[0].set_title("Training loss vs epoch")

axs[1].plot(np.arange(epochs),train_acc, label="Train acc")
axs[1].plot(np.arange(epochs),val_acc, label="Val acc")
axs[1].legend()
axs[1].set_xlabel("Epoch")
axs[1].set_ylabel("Accuracy")
axs[1].set_title("Training and validation accuracy vs epoch")

plt.show()
#endregion
#region hyperparameter search
# %%
args = {
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'hidden_size': 32,
    'weight_decay': 1e-5,
    'lr': 0.001,
    'epochs':100,
    'k_hits':[5,10,20,50]
}
# %%
# %%
def opt_iteration(G,edge_train_mode,supervision_mode,feature_length,init_mode,args):
  """
  graph: networkx graph object
  edge_train_mode: "disjoint" or "all", supervision_mode: "all" or "gda_only", feature_length: int,
  init_mode:"random" or "ones"
  args: arguments to be shared by all iterations
  """
  results = {}
  epochs = args["epochs"]
  hits_range = args["k_hits"]
  init_node_features(G,init_mode,feature_length)
  Hete = HeteroGraph(G)

  dataset = GraphDataset([Hete], task="link_pred", edge_train_mode=edge_train_mode, resample_negatives=True)
  dataset_train, dataset_val, dataset_test = dataset.split(transductive=True, split_ratio=[0.4, 0.3, 0.3])
  splits = {"train":dataset_train[0].to(args["device"]), "val":dataset_val[0].to(args["device"]), "test":dataset_test[0].to(args["device"])}

  model = HeteroGNN(Hete,"dotprod", supervision_mode, args, aggr="mean",).to(args["device"])
  optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
  loss_plot = []
  train_acc = []
  val_acc = []
  for epoch in range(epochs):
    if epoch%10 == 0:
       loss,t_acc = train(model,optimizer,splits["train"],printb=True)
       v_acc = test2(model,splits["val"])
    else:
        loss,t_acc = train(model,optimizer,splits["train"],printb=False)
        v_acc = test2(model,splits["val"])
    loss_plot.append(loss)
    train_acc.append(t_acc)
    val_acc.append(v_acc)

  final_acc_train = test2(model,splits["train"])
  final_acc_val = test2(model,splits["val"])
  results["Train acc"] = final_acc_train
  results["Validation acc"] = final_acc_val

  hit_dict = {}
  for k in hits_range:
    hits = hits_at_k(model,splits["val"],k)
    hit_dict[k] = hits
  
  results["Hits @ k"] = hit_dict
  edgetype_acc = test2(model,splits["val"],global_accuracy=False)
  results["Type Accuracy"] = edgetype_acc

  fig,axs = plt.subplots(1, 2, figsize=(18,6))
  fig.suptitle(f"Dimensión de feature:{feature_length}, modo de supervisión: {supervision_mode}", fontsize=16)
  axs[0].plot(np.arange(epochs),loss_plot)
  axs[0].set_xlabel("Epoch",fontsize=14)
  axs[0].set_ylabel("Training Loss",fontsize=14)
  axs[0].set_title("Training loss vs epoch", fontsize=16)

  axs[1].plot(np.arange(epochs),train_acc, label="Train acc")
  axs[1].plot(np.arange(epochs),val_acc, label="Val acc")
  axs[1].legend()
  axs[1].set_xlabel("Epoch",fontsize=14)
  axs[1].set_ylabel("Accuracy (Global)",fontsize=14)
  axs[1].set_title("Training, validation accuracy vs epoch", fontsize=16)

  #{model name}_{feature_mode}_{feature length}_{supervision mode}
  plt.savefig(f"/content/figures/rsage_{init_mode}_{feature_length}_{supervision_mode}.png")
  plt.show()
  
  torch.cuda.empty_cache()
  return results

# %%
torch.cuda.empty_cache()

# %%
import pickle

def save_results(result_dict,filename):
    with open(filename +'.pickle', 'wb') as f:
        pickle.dump(result_dict, f, pickle.HIGHEST_PROTOCOL)

# %%
all_results = {}
supervision = ["all","gda_only"]
feature_len = [5,10,50,100,500]
for sup in supervision:
  for flen in feature_len:
    result = opt_iteration(G,"disjoint",sup,flen,"random",args)
    all_results[(sup,flen)] = result

save_results(all_results,"/content/results_5_7")

# %%
save_results(all_results,"/content/results_5_7")

# %%
with open('results_5_7.pickle', 'rb') as handle:
    b = pickle.load(handle)

# %%
b

# %% [markdown]
# # aver

# %%
hits = hits_at_k(model,splits["val"],5)

# %%
splits["val"].edge_label_index[('disease', 'GDA', 'gene/protein')]

# %%
hits

# %%
index = torch.tensor([1,2]).to(args["device"])
torch.index_select(splits["val"].edge_label_index[('disease', 'GDA', 'gene/protein')],-1,index)

# %%
# %%
for edge in predicted_edges:
  #(f"Edge ({edge[0]}, {edge[1]}) has properties:", edge[2])
  print(f"Node {edge[0]} has properties:", G.nodes(data=True)[edge[0]])
  print(f"Node {edge[1]} has properties:", G.nodes(data=True)[edge[1]])

# %%
predicted_index = hits[('disease', 'GDA', 'gene/protein')]["indices"]
predicted_edges = torch.index_select(splits["val"].edge_label_index[('disease', 'GDA', 'gene/protein')],-1,predicted_index)
edgelist = tensor_to_edgelist(predicted_edges)
for edge in edgelist:
  #(f"Edge ({edge[0]}, {edge[1]}) has properties:", edge[2])
  print(f"Node {edge[0]} has properties:", G.nodes(data=True)[edge[0]])
  print(f"Node {edge[1]} has properties:", G.nodes(data=True)[edge[1]])

# %%
edgelist
#endregion
#region explore predictions
# %%
prediction = get_prediction(model,splits["train"])
train_gene_data = get_node_data(splits["train"],"gene/protein")
train_disease_data = get_node_data(splits["train"],"disease")
train_complex_data = get_node_data(splits["train"],"protein_complex")
train_nodes_data = {"disease":train_disease_data, "gene/protein":train_gene_data, "protein_complex":train_complex_data}
aver = map_prediction_to_edges(prediction,splits["train"])
PC_predictions = aver[("gene/protein","forms_complex","protein_complex")]
#endregion

#region try to fix split
#%%
def init_node_feature_map(G, mode, size):
  """ Para usar con nx.Graph """
  if mode == "ones":
    feature = torch.ones(size)
    feature_dict = {node:feature for node in list(G.nodes())}
  elif mode == "random":
    feature_dict = {}
    for node in list(G.nodes()):
      feature_dict[node] = torch.rand(size)
  return feature_dict

def generate_features(G,modes_sizes:dict) -> dict:
  feature_dict = {}
  for mode,sizes in modes_sizes.items():
    for size in sizes:
      feature_map = init_node_feature_map(G,mode,size)
      feature_dict[(mode,size)] = feature_map
  return feature_dict

def init_multiple_features(G,modes_sizes:dict) -> dict:
  for mode,sizes in modes_sizes.items():
    for size in sizes:
      feature_map = init_node_feature_map(G,mode,size)
      nx.set_node_attributes(G,feature_map,f"{mode}_{size}")

init_multiple_features(G,{"ones":[1,2,3],"random":[1,2,3,4]})
Hete = HeteroGraph(G)

#TODO: escribir test para ver si las features son las mismas en todos los splits. Esto funca para hacerlo a ojo un par de veces
count = torch.sum(splits["train"].key['gene/protein'] != splits["val"].ones_3['gene/protein'] != splits["test"]).item()
#%%
#region try to change feature vector post split

#placeholder feature:

#inicializo features
nx.set_node_attributes(G,[],"node_feature")
init_multiple_features(G,{"ones":[1,2,3],"random":[1,2,3,4]})
Hete = HeteroGraph(G)

#verifico que funciona
list(G.nodes(data=True))[0:2]
Hete.keys
Hete.random_1

#hago el split y vuelvo a verificar
args = {
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'hidden_size': 32,
    'weight_decay': 1e-5,
    'lr': 0.001,
}
task = 'link_pred'
train_mode = "disjoint"

dataset = GraphDataset([Hete], task=task, edge_train_mode=train_mode, resample_negatives=True)
dataset_train, dataset_val, dataset_test = dataset.split(transductive=True, split_ratio=[0.4, 0.3, 0.3])
splits = {"train":dataset_train[0].to(args["device"]), "val":dataset_val[0].to(args["device"]), "test":dataset_test[0].to(args["device"])}

#TODO: escribir test para automatizar esto y estar segura de que no falla en ningun caso
list(splits["train"].G.nodes(data=True))[0:2]
list(splits["val"].G.nodes(data=True))[0:2]
list(splits["test"].G.nodes(data=True))[0:2]

splits["train"].random_3
splits["val"].random_3
splits["test"].random_3

#una verificación medio a ojo que habría que generalizar a un test
count = torch.sum(splits["train"].random_3['gene/protein'] != splits["val"].random_3['gene/protein']).item()

#listo, funciona, ahora a tratar de modificar el feature vector en cada iteración
splits["train"].node_feature
Hete.__setattr__("node_feature",Hete.random_1)
Hete.node_feature 
#funciona!!!

splits["val"].__setattr__("node_feature",splits["val"].random_3)
splits["train"].node_feature

#ahora a modificar el training loop para incluir esto
def set_feature(split,init_mode,feature_length):
  feature_key = f"{init_mode}_{feature_length}"
  split.__setattr__("node_feature",split[feature_key])


#%%
##### ACÁ PRIMERO DEBERÍA PONER EL SPLIT Y LA INICIALIZACIÓN DE FEATURES ####


def opt_iteration2(H:HeteroGraph,train_set:HeteroGraph,val_set:HeteroGraph,edge_train_mode:str,supervision_mode:str,decoder_head:str,feature_length:int,init_mode:str,args:dict):
  """
  graph: networkx graph object
  edge_train_mode: "disjoint" or "all", supervision_mode: "all" or "gda_only", feature_length: int,
  init_mode:"random" or "ones"
  args: arguments to be shared by all iterations
  """
  #select feature que vamos a usar
  set_feature(H,init_mode,feature_length)
  set_feature(train_set,init_mode,feature_length)
  set_feature(val_set,init_mode,feature_length)

  results = {}
  epochs = args["epochs"]
  hits_range = args["k_hits"]

  model = HeteroGNN(H,decoder_head, supervision_mode, args, aggr="mean",).to(args["device"])
  optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
  loss_plot = []
  train_acc = []
  val_acc = []
  for epoch in range(epochs):
    if epoch%10 == 0:
       loss,t_acc = train(model,optimizer,train_set,printb=True)
       v_acc = test2(model,val_set)
    else:
        loss,t_acc = train(model,optimizer,train_set,printb=False)
        v_acc = test2(model,val_set)
    loss_plot.append(loss)
    train_acc.append(t_acc)
    val_acc.append(v_acc)

  final_acc_train = test2(model,train_set)
  final_acc_val = test2(model,val_set)
  results["Train acc"] = final_acc_train
  results["Validation acc"] = final_acc_val

  hit_dict = {}
  for k in hits_range:
    hits = hits_at_k(model,val_set,k)
    hit_dict[k] = hits
  
  results["Hits @ k"] = hit_dict
  edgetype_acc = test2(model,val_set,global_accuracy=False)
  results["Type Accuracy"] = edgetype_acc

  fig,axs = plt.subplots(1, 2, figsize=(18,6))
  fig.suptitle(f"Dimensión de feature:{feature_length}, modo de supervisión: {supervision_mode}", fontsize=16)
  axs[0].plot(np.arange(epochs),loss_plot)
  axs[0].set_xlabel("Epoch",fontsize=14)
  axs[0].set_ylabel("Training Loss",fontsize=14)
  axs[0].set_title("Training loss vs epoch", fontsize=16)

  axs[1].plot(np.arange(epochs),train_acc, label="Train acc")
  axs[1].plot(np.arange(epochs),val_acc, label="Val acc")
  axs[1].legend()
  axs[1].set_xlabel("Epoch",fontsize=14)
  axs[1].set_ylabel("Accuracy (Global)",fontsize=14)
  axs[1].set_title("Training, validation accuracy vs epoch", fontsize=16)

  #{model name}_{feature_mode}_{feature length}_{supervision mode}
  plt.savefig(f"/content/figures/rsage_{init_mode}_{feature_length}_{supervision_mode}.png")
  plt.show()
  
  torch.cuda.empty_cache()
  return results

#%%

args = {
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'hidden_size': 32,
    'weight_decay': 1e-5,
    'lr': 0.001,
    'epochs':100,
    'k_hits':[5,10,20,50]
}

opt_iteration2(Hete,splits["train"],splits["val"],"disjoint","all","dotprod",3,"random",args)
splits["train"].node_feature
#%%
#init_node_features(G,"random",3)
Hete = HeteroGraph(G)
Hete.node_feature
splits["train"].node_feature
splits["val"].node_feature