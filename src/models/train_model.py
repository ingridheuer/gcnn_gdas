#%%
import torch
import random
import config
import os
import datetime
import pickle
import training_utils
import sage_ones
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

seed = config.train_config["misc"]["seed"]
random.seed(seed)
torch.manual_seed(seed)
#%%
# Load data and configure results folder
# Datasets are already split
data_folder = config.train_config["data"]["dataset_folder_path"]
save_to_path = config.train_config["data"]["results_folder_path"]

if not os.path.exists(save_to_path):
    print("save_to dir does not exist, a new directory will be created")
    os.makedirs(save_to_path)


train_data,val_data = training_utils.load_data(data_folder)

# Initialize features
feature_type = config.train_config["features"]["feature_type"]
feature_dim = config.train_config["features"]["feature_dim"]

if feature_type != "natural":
    train_data = training_utils.initialize_features(train_data,feature_type,feature_dim)
    val_data = training_utils.initialize_features(val_data,feature_type,feature_dim)

# Initialize model
# TODO: add "all types supervision"
supervision_type_map = {"gda":("gene_protein","gda","disease"),"disease_disease":("disease","disease_disease","disease")}
supervision_types = config.train_config["model"]["supervision_types"]
supervision_types_mapped = [supervision_type_map[edge_type] for edge_type in supervision_types]

model_type = config.train_config["model"]["model"]
if model_type == "sage_ones":
    model = sage_ones.Model(train_data.metadata(),supervision_types_mapped)

# Train model
metric_map = {"roc_auc":roc_auc_score, "average_precision":average_precision_score, "accuracy":accuracy_score}

def train_model(model,train_set,val_set,params,plot_title=f"Training {model_type}"):
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params["weight_decay"])
    train_losses = []
    val_losses = []
    train_scores = []
    val_scores = []

    metric = metric_map[params["metric"]]
    epochs = params["epochs"]

    early_stopper = training_utils.EarlyStopper(params["patience"],params["delta"])
    for epoch in range(epochs):
        train_loss = training_utils.train(model,optimizer,train_set)
        train_score = training_utils.test(model,train_set,metric)

        val_loss = training_utils.get_val_loss(model,val_set)
        val_score = training_utils.test(model,val_set,metric)

        train_losses.append(train_loss)
        train_scores.append(train_score)

        val_scores.append(val_score)
        val_losses.append(val_loss)

        if epoch%50 == 0:
            print(train_loss)
        
        if early_stopper.early_stop(val_loss):
            print("Early stopping")
            break

    val_auc = training_utils.test(model,val_set,roc_auc_score)
    curve_data = [train_losses,val_losses,train_scores,val_scores]

    training_utils.plot_training_stats(plot_title, *curve_data,"AUC")
    return model, val_auc, curve_data

#%%
training_params = config.train_config["train_params"]
plot_title = config.train_config["misc"]["plot_title"]

trained_model, val_auc, curve_data = train_model(model,train_data,val_data,training_params,plot_title)
#%%
model_name = config.train_config["misc"]["model_name"]
date = datetime.datetime.now()
fdate = date.strftime("%d_%m_%y__%H_%M")
fname = f"{save_to_path}{model_name}_{fdate}"

if config.train_config["misc"]["save_trained_model"]:
    torch.save(trained_model.state_dict(), f"{fname}.pth")

if config.train_config["misc"]["save_plot_data"]:
    with open(f"{fname}.pkl", 'wb') as f:
        pickle.dump(curve_data, f)

# %%
