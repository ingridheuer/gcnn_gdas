#%%
import torch
from deepsnap.hetero_graph import HeteroGraph
from deepsnap.dataset import GraphDataset
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

import convolutions
import utils
#%%
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

def test(model,validation_set,global_accuracy=True):
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

def hits_at_k(model,dataset,k,args) -> dict:
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
        hits[key] = labels.sum().item()

  return hits

def optimization_iteration(H:HeteroGraph,train_set:HeteroGraph,val_set:HeteroGraph,edge_train_mode:str,supervision_mode:str,decoder_head:str,feature_length:int,init_mode:str,args:dict,reports_path:str):
  """
  graph: networkx graph object
  edge_train_mode: "disjoint" or "all", supervision_mode: "all" or "gda_only", feature_length: int,
  init_mode:"random" or "ones"
  args: arguments to be shared by all iterations
  """
  #select feature que vamos a usar
  utils.set_feature(H,init_mode,feature_length)
  utils.set_feature(train_set,init_mode,feature_length)
  utils.set_feature(val_set,init_mode,feature_length)
  train_set.to(args["device"])
  val_set.to(args["device"])

  results = {}
  epochs = args["epochs"]
  hits_range = args["k_hits"]

  model = convolutions.HeteroGNN(H,decoder_head, supervision_mode, args, aggr="mean",).to(args["device"])
  optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
  loss_plot = []
  train_acc = []
  val_acc = []
  for epoch in range(epochs):
    if epoch%10 == 0:
       loss,t_acc = train(model,optimizer,train_set,printb=True)
       v_acc = test(model,val_set)
    else:
        loss,t_acc = train(model,optimizer,train_set,printb=False)
        v_acc = test(model,val_set)
    loss_plot.append(loss)
    train_acc.append(t_acc)
    val_acc.append(v_acc)

  final_acc_train = test(model,train_set)
  final_acc_val = test(model,val_set)
  results["Train acc"] = final_acc_train
  results["Validation acc"] = final_acc_val

  hit_dict = {}
  for k in hits_range:
    hits = hits_at_k(model,val_set,k,args)
    hit_dict[k] = hits
  
  results["Hits @ k"] = hit_dict
  edgetype_acc = test(model,val_set,global_accuracy=False)
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
  plt.savefig(f"{reports_path}rsage_{init_mode}_{feature_length}_{supervision_mode}.png")
  plt.show()
  
  torch.cuda.empty_cache()
  return results,model

#%%
graph_data = nx.read_gml("../../data/processed/graph_data_nohubs/processed_graph.gml")

G = graph_data.to_directed()
#placeholder feature:
nx.set_node_attributes(G,[],"node_feature")
#inicializo features
utils.init_multiple_features(G,{"random":[5]})

utils.check_symmetry_test(G)
Hete = HeteroGraph(G)

utils.set_feature(Hete,"random",5)

task = 'link_pred'

args = {
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'hidden_size': 32,
    'weight_decay': 1e-5,
    'lr': 0.001,
    'epochs':50,
    'k_hits':[50,200]
}

model = convolutions.HeteroGNN(Hete,"dotprod","gda_only",args)
#%%
model(Hete)
#%%