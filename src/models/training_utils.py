import torch
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import copy
import datetime
import pandas as pd
import pickle 
import sys
sys.path.append("..")

@torch.no_grad()
def hits_at_k(y_true,x_prob,k,key) -> dict:
    """Dados los tensores x_prob y edge_label, calcula cuantas predicciones hizo correctamente en los primeros k puntajes.
    x_prob es la predicción del modelo luego de aplicar sigmoid (sin redondear, osea, el puntaje crudo)"""

    #ordeno los puntajes de mayor a menor
    x_prob, indices = torch.sort(x_prob, descending=True)

    #me quedo solo con los k mayor punteados
    x_prob = x_prob[:k]
    indices = indices[:k]

    if any(x_prob < 0.5):
      threshold_index = (x_prob < 0.5).nonzero()[0].item()
      print(f"Top {k} scores for {key} below classification threshold 0.5, threshold index: {threshold_index}")

    #busco que label tenían esas k preds
    labels = y_true[indices]

    #cuento cuantas veces predije uno positivo en el top k
    hits = labels.sum().item()

    return hits

def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    preds = model(data.x_dict,data.adj_t_dict,data.edge_label_index_dict)
    edge_label = data.edge_label_dict
    loss = model.loss(preds, edge_label)
    loss.backward()
    optimizer.step()

    return loss.item()

@torch.no_grad()
def get_val_loss(model,val_data):
    model.eval()
    preds = model(val_data.x_dict,val_data.adj_t_dict,val_data.edge_label_index_dict)
    edge_label = val_data.edge_label_dict
    loss = model.loss(preds, edge_label)

    return loss.item()

def get_metrics(y_true, x_pred):
   acc = round(accuracy_score(y_true,x_pred),2)
   ap = round(average_precision_score(y_true, x_pred),2)
   roc_auc = round(roc_auc_score(y_true,x_pred),2)

   return acc,ap ,roc_auc
  
@torch.no_grad()
def test(model,data,metric):
  model.eval()
  preds = model(data.x_dict,data.adj_t_dict,data.edge_label_index_dict)
  edge_label = data.edge_label_dict
  all_preds = []
  all_true = []
  for key,pred in preds.items():
      pred_label = torch.round(pred)
      ground_truth = edge_label[key]
      all_preds.append(pred_label)
      all_true.append(ground_truth)
  total_predictions = torch.cat(all_preds, dim=0).cpu().numpy()
  total_true = torch.cat(all_true, dim=0).cpu().numpy()
  score = metric(total_true,total_predictions)
  return score
  

@torch.no_grad()
def full_test(model,data,k,global_score=True):
  model.eval()
  preds = model(data.x_dict,data.adj_t_dict,data.edge_label_index_dict)
  edge_label = data.edge_label_dict
  metrics = {}

  if global_score:
    all_scores = []
    all_preds = []
    all_true = []
    for key,pred in preds.items():
        pred_label = torch.round(pred)
        ground_truth = edge_label[key]
        all_scores.append(pred)
        all_preds.append(pred_label)
        all_true.append(ground_truth)

    total_predictions = torch.cat(all_preds, dim=0)
    total_true = torch.cat(all_true, dim=0)
    total_scores = torch.cat(all_scores,dim=0)

    acc, ap, roc_auc =  get_metrics(total_true.cpu().numpy(), total_predictions.cpu().numpy())
    hits_k = hits_at_k(total_true,total_scores,k,"all")
    metrics["all"] = [acc,ap,roc_auc,hits_k]

  else:
    for key,pred in preds.items():
        pred_label = torch.round(pred)
        ground_truth = edge_label[key]
        acc, ap, roc_auc = get_metrics(ground_truth.cpu().numpy(), pred_label.cpu().numpy())
        hits_k = hits_at_k(ground_truth,pred,k,key)
        metrics[key] = [acc,ap, roc_auc,hits_k]
  
  return metrics

def plot_training_stats(title, train_losses,val_losses, train_metric,val_metric,metric_str):

  fig, ax = plt.subplots(figsize=(8,5))
  ax2 = ax.twinx()

  ax.set_xlabel("Training Epochs")
  ax2.set_ylabel(metric_str)
  ax.set_ylabel("Loss")

  plt.title(title)
  p1, = ax.plot(train_losses, "b-", label="training loss")
  p2, = ax2.plot(val_metric, "r-", label=f"val {metric_str}")
  p3, = ax2.plot(train_metric, "o-", label=f"train {metric_str}")
  p4, = ax.plot(val_losses,"b--",label=f"validation loss")
  plt.legend(handles=[p1, p2, p3,p4],loc=2)
  plt.show()

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def launch_experiment(model,train_set,val_set,params,plot_title="title"):
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params["weight_decay"])
    train_losses = []
    val_losses = []
    train_scores = []
    val_scores = []

    metric = roc_auc_score
    epochs = params["epochs"]

    early_stopper = EarlyStopper(params["patience"],params["delta"])
    for epoch in range(epochs):
        train_loss = train(model,optimizer,train_set)
        val_loss = get_val_loss(model,val_set)
        train_score = test(model,train_set,metric)
        val_score = test(model,val_set,metric)

        train_losses.append(train_loss)
        train_scores.append(train_score)
        val_scores.append(val_score)
        val_losses.append(val_loss)

        if epoch%50 == 0:
            print(train_loss)
        
        if early_stopper.early_stop(val_loss):
            print("Early stopping")
            break

    val_auc = test(model,val_set,roc_auc_score)
    curve_data = [train_losses,val_losses,train_scores,val_scores]

    plot_training_stats(plot_title, *curve_data,"AUC")
    return model, val_auc

def load_data(folder_path,load_test = False):
    if load_test:
        names = ["train","validation","test"]
    else:
        names = ["train","validation"]
    datasets = []
    for name in names:
        path = folder_path+name+".pt"
        datasets.append(torch.load(path))
    
    with open(folder_path+"node_map.pickle", 'rb') as handle:
        node_map = pickle.load(handle)
    
    return datasets, node_map

def initialize_features(data,feature,dim,inplace=False):
    if inplace:
        data_object = data
    else:
        data_object = copy.copy(data)
    for nodetype, store in data_object.node_items():
        if feature == "random":
            data_object[nodetype].x = torch.rand(store["num_nodes"],dim)
        if feature == "ones":
            data_object[nodetype].x = torch.ones(store["num_nodes"],dim)
    return data_object

def save_model(model,folder_path,model_name):
    date = datetime.datetime.now()
    fdate = date.strftime("%d_%m_%y__%H_%M_%S")
    fname = f"{model_name}_{fdate}"
    torch.save(model.state_dict(), f"{folder_path}{fname}.pth")

def load_model(weights_path,model_type,supervision_types,metadata,model_args=None):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = torch.load(weights_path,map_location=torch.device(device))

    if model_type == "base_model":
        from models import base_model
        model = base_model.base_model(model_args,metadata,supervision_types)
    elif model_type == "sage_ones":
        from models import sage_ones
        model = sage_ones.Model(metadata,supervision_types)
    
    model.load_state_dict(weights)
    
    return model

@torch.no_grad()
def get_encodings(model,data):
    model.eval()
    x_dict = data.x_dict
    edge_index = data.edge_index_dict
    encodings = model.encoder(x_dict,edge_index)
    return encodings

def load_node_csv(path, index_col,type_col, **kwargs):
    """Returns node dataframe and a dict of mappings for each node type. 
    Each mapping maps from original df index to "heterodata index" { node_type : { dataframe_index : heterodata_index}}"""
    df = pd.read_csv(path, **kwargs,index_col=index_col)
    node_types = df[type_col].unique()
    mappings_dict = dict()
    for node_type in node_types:
        mapping = {index: i for i, index in enumerate(df[df[type_col] == node_type].index.unique())}
        mappings_dict[node_type] = mapping

    return df,mappings_dict

class NegativeSampler:
    def __init__(self,full_dataset,edge_type,src_degrees,dst_degrees) -> None:
        src_type, _ , dst_type = edge_type
        self.num_nodes = (full_dataset[src_type]["num_nodes"],full_dataset[dst_type]["num_nodes"])

        full_positive_index = full_dataset.edge_index_dict[edge_type]
        self.full_positive_hash = self.index_to_hash(full_positive_index)

        self.weights = [src_degrees,dst_degrees]
    
    def index_to_hash(self,edge_index):
        size = self.num_nodes
        row, col = edge_index
        hashed_edges = (row * size[1]).add_(col)
        return hashed_edges

    def hash_to_index(self,hashed_edges):
        size = self.num_nodes
        row = hashed_edges.div(size[1], rounding_mode='floor')
        col = hashed_edges % size[1]
        return torch.stack([row, col], dim=0)
    
    def sample_negatives(self,num_samples,src_or_dst):
        """num_samples: number of samples generated, output will have shape [num_samples]. 
        src_or_dst: use src or dst weights to generate sample. 0:src weights, 1:dst weights
        """
        probs = torch.tensor(self.weights[src_or_dst]**0.75)
        neg_samples = probs.multinomial(num_samples, replacement=True)
        return neg_samples
    
    def generate_negative_edge_index(self,positive_edge_index,method):
        if method == "corrupt_both":
            num_samples = positive_edge_index.shape[1]
            new_src_index = self.sample_negatives(num_samples,0)
            new_dst_index = self.sample_negatives(num_samples,1)
            negative_edge_index = torch.stack([new_src_index,new_dst_index])
            return negative_edge_index
        elif method == "fix_src":
            src_index, _ = positive_edge_index
            new_dst_index = self.sample_negatives(src_index.numel(),1)
            negative_edge_index = torch.stack([src_index,new_dst_index])
            return negative_edge_index
        elif method == "fix_dst":
            _, dst_index = positive_edge_index
            new_src_index = self.sample_negatives(dst_index.numel(),0)
            negative_edge_index = torch.stack([new_src_index,dst_index])
            return negative_edge_index            
    
    def test_false_negatives(self,negative_edge_index,positive_edge_index):
        full_hash = self.full_positive_hash
        negative_hash = self.index_to_hash(negative_edge_index)
        positive_hash = self.index_to_hash(positive_edge_index)

        false_negatives_mask = torch.isin(negative_hash,full_hash)
        new_negative_hash = negative_hash[~false_negatives_mask]
        retry_positive_hash = positive_hash[false_negatives_mask]

        return new_negative_hash, retry_positive_hash
    
    def get_negative_sample(self,positive_edge_index,method):
        true_negatives = []
        retry_positive_hash = torch.tensor([0]) #placeholder
        temp_positive_edge_index = copy.copy(positive_edge_index)

        while retry_positive_hash.numel() > 0:
            negative_edge_index = self.generate_negative_edge_index(temp_positive_edge_index,method)
            true_neg_hash, retry_positive_hash = self.test_false_negatives(negative_edge_index,temp_positive_edge_index)

            true_negatives.append(true_neg_hash)
            temp_positive_edge_index = self.hash_to_index(retry_positive_hash)


        negative_edge_hash = torch.concat(true_negatives)
        negative_edge_index = self.hash_to_index(negative_edge_hash)

        return negative_edge_index
    
    def get_labeled_tensors(self,positive_edge_index,method):
        """positive_edge_index: edge_index with only positive edges. 
        This function will use positive_edge_index as a starting point to generate a negative index
        with the same shape as positive_edge_index.
        
        method: 
        corrupt_both: sample both src and dst nodes with probability deg**0.75
        fix_src: keep original src nodes fixed and sample dst nodes with probability deg**0.75
        fix_dst: like fix_src but keep original dst nodes"""

        sample = self.get_negative_sample(positive_edge_index,method)
        edge_label_index = torch.concat([positive_edge_index,sample],dim=1)
        edge_label = torch.concat([torch.ones(positive_edge_index.shape[1]), torch.zeros(positive_edge_index.shape[1])])
        return edge_label_index, edge_label


def get_tensor_index_df(node_data, node_map, node_info):
    sub_dfs = []
    for node_type in node_map.keys():
        sub_df = node_data[node_data.node_type == node_type]
        node_map_series = pd.Series(node_map[node_type], name="tensor_index")
        sub_df = sub_df.merge(node_map_series, left_on="node_index", right_index=True,
                              how="right").sort_values(by="tensor_index").reset_index()

        sub_dfs.append(sub_df)
    tensor_df = pd.concat(sub_dfs, ignore_index=True)
    df = pd.merge(tensor_df, node_info[["node_index", "comunidades_infomap", "comunidades_louvain",
                  "degree_gda", "degree_pp", "degree_dd"]], on="node_index")
    df["total_degree"] = df.degree_pp + df.degree_gda + df.degree_dd
    return df