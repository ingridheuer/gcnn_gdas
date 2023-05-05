import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
import matplotlib.pyplot as plt

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

def train(model, optimizer, graph,supervision_types):
    model.train()
    optimizer.zero_grad()
    preds = model(graph,supervision_types)
    edge_label = graph.edge_label_dict
    loss = model.loss(preds, edge_label)
    loss.backward()
    optimizer.step()

    return loss.item()

@torch.no_grad()
def get_val_loss(model,val_data,supervision_types):
    model.eval()
    preds = model(val_data,supervision_types)
    edge_label = val_data.edge_label_dict
    loss = model.loss(preds, edge_label)

    return loss.item()

def get_metrics(y_true, x_pred):
   acc = round(accuracy_score(y_true,x_pred),2)
   ap = round(average_precision_score(y_true, x_pred),2)
   roc_auc = round(roc_auc_score(y_true,x_pred),2)

   return acc,ap ,roc_auc
  
@torch.no_grad()
def test(model,data,supervision_types,metric):
  model.eval()
  preds = model(data,supervision_types)
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
def full_test(model,data,supervision_types,k,global_score=True):
  model.eval()
  preds = model(data,supervision_types)
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

