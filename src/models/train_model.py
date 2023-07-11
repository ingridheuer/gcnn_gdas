#%%
import torch
import config
import os
import datetime
import pickle
import training_utils
from torch_geometric import seed_everything
import sage_ones

seed = config.train_config["misc"]["seed"]
seed_everything(seed)
#%%
# Load data and configure results folder
# Datasets are already split
data_folder = config.train_config["data"]["dataset_folder_path"] + f"seed_{seed}/"
save_to_path = config.train_config["data"]["results_folder_path"]

if not os.path.exists(save_to_path):
    print("save_to dir does not exist, a new directory will be created")
    os.makedirs(save_to_path)

datasets, node_map = training_utils.load_data(data_folder)
train_data, val_data = datasets
full_dataset = torch.load(data_folder+"full_dataset.pt")

# Initialize features
feature_type = config.train_config["features"]["feature_type"]
feature_dim = config.train_config["features"]["feature_dim"]

if feature_type != "lsa":
    train_data = training_utils.initialize_features(train_data,feature_type,feature_dim)
    val_data = training_utils.initialize_features(val_data,feature_type,feature_dim)
else: 
    train_data = training_utils.initialize_features(train_data,feature_type,feature_dim,data_folder)
    val_data = training_utils.initialize_features(val_data,feature_type,feature_dim,data_folder)

# Initialize model
# TODO: add "all types supervision"
supervision_type_map = {"gda":("gene_protein","gda","disease"),"disease_disease":("disease","disease_disease","disease")}
supervision_types = config.train_config["model"]["supervision_types"]
supervision_types_mapped = [supervision_type_map[edge_type] for edge_type in supervision_types]

model_type = config.train_config["model"]["model"]
if model_type == "sage_ones":
    model = sage_ones.Model(train_data.metadata(),supervision_types_mapped)
else:
    print("Invalid model type")

# Train model

def train_model(model,train_set,val_set,full_set,params,sample_epochs,sample_ratio,plot_title=f"Training {model_type}"):
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params["weight_decay"])
    train_losses = []
    val_losses = []
    train_scores = []
    val_scores = []

    epochs = params["epochs"]

    early_stopper = training_utils.EarlyStopper(params["patience"],params["delta"])
    negative_sampler = training_utils.NegativeSampler(full_set,("gene_protein","gda","disease"),full_set["gene_protein"]["degree_gda"],full_set["disease"]["degree_gda"])
    train_label_index = train_set["gene_protein","gda","disease"]["edge_label_index"]
    for epoch in range(epochs):
        #Resample supervision links every k epochs
        if epoch%sample_epochs == 0:
            sample_index = torch.randint(high=train_label_index.shape[1], size=(round(sample_ratio*train_label_index.shape[1]),))
            positive_sample = train_label_index[:,sample_index]

            # positive_sample = train_label_index
            new_train_label_index, new_train_label = negative_sampler.get_labeled_tensors(positive_sample,"corrupt_both")
            train_set["gene_protein","gda","disease"]["edge_label_index"] = new_train_label_index
            train_set["gene_protein","gda","disease"]["edge_label"] = new_train_label

        train_loss = training_utils.train(model,optimizer,train_set)
        train_score = training_utils.test(model,train_set)

        val_loss = training_utils.get_val_loss(model,val_set)
        val_score = training_utils.test(model,val_set)

        train_losses.append(train_loss)
        train_scores.append(train_score)

        val_scores.append(val_score)
        val_losses.append(val_loss)

        if epoch%50 == 0:
            print(round(train_loss,2))
        
        if early_stopper.early_stop(val_loss):
            print("Early stopping")
            break

    val_auc = training_utils.test(model,val_set)
    curve_data = [train_losses,val_losses,train_scores,val_scores]

    training_utils.plot_training_stats(plot_title, *curve_data,"AUC")
    return model, val_auc, curve_data

#%%
training_params = config.train_config["train_params"]
plot_title = config.train_config["misc"]["plot_title"]

trained_model, val_auc, curve_data = train_model(model,train_data,val_data,full_dataset,training_params,10,0.8,plot_title)
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
training_utils.full_test(trained_model,val_data,200,False)
# %%