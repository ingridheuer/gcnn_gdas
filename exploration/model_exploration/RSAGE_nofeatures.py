# %%
import torch
from deepsnap.dataset import GraphDataset
from deepsnap.hetero_graph import HeteroGraph
#from torch_geometric.data import HeteroData

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import convolutions
import utils
import prediction_heads
# %%
debugging = False

def my_debug(arg):
    if debugging:
        print(arg)
#%%

G = nx.read_gml("../../data/processed/graph_data_nohubs/processed_graph.gml")

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

# init_multiple_features(G,{"ones":[1,2,3],"random":[1,2,3,4]})
Hete = HeteroGraph(G)


dataset = GraphDataset([Hete], task=task, edge_train_mode=train_mode, resample_negatives=True)
dataset_train, dataset_val, dataset_test = dataset.split(transductive=True, split_ratio=[0.4, 0.3, 0.3])
splits = {"train":dataset_train[0].to(args["device"]), "val":dataset_val[0].to(args["device"]), "test":dataset_test[0].to(args["device"])}
#endregion
#region init model
# %%
model = convolutions.HeteroGNN(Hete,"dotprod", "all", args, aggr="mean",).to(args["device"])
optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
#endregion
#region train model
# %%
loss_plot = []
train_acc = []
val_acc = []
epochs = 1
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