#%%
import torch
import numpy as np
from torch_geometric import seed_everything
import final_model, training_utils
import pickle
seed_everything(0)
#%%
data_folder = "../../data/processed/graph_data_nohubs/merged_types/split_dataset/"
models_folder = "../../models/final_model/"
feature_folder = "../../data/processed/feature_data/"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#%%
def run_experiment(params, train_set, val_set,negative_sampler,feature_folder=feature_folder):
    # Initialize node features
    train_set = training_utils.initialize_features(train_set, params["feature_type"], params["feature_dim"], feature_folder)
    val_set = training_utils.initialize_features(val_set, params["feature_type"],params["feature_dim"], feature_folder)

    train_set.to(device)
    val_set.to(device)

    # Initialize model
    model = final_model.Model(train_set.metadata(),[("gene_protein","gda","disease")])
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"]
    )
    train_losses = []
    val_losses = []
    train_scores = []
    val_scores = []

    early_stopper = training_utils.EarlyStopper(params["patience"], params["delta"])
    train_label_index = train_set["gene_protein","gda","disease"]["edge_label_index"]

    for epoch in range(params["epochs"]):
        #Resample negative supervision links every epoch
        new_train_label_index, new_train_label = negative_sampler.get_labeled_tensors(train_label_index.cpu(),"corrupt_both")
        train_set["gene_protein","gda","disease"]["edge_label_index"] = new_train_label_index.to(device)
        train_set["gene_protein","gda","disease"]["edge_label"] = new_train_label.to(device)

        train_loss = training_utils.train(model, optimizer, train_set)
        val_loss = training_utils.get_val_loss(model, val_set)

        train_score = training_utils.test(model, train_set)
        val_score = training_utils.test(model, val_set)

        train_losses.append(train_loss)
        train_scores.append(train_score)

        val_scores.append(val_score)
        val_losses.append(val_loss)

        if early_stopper.early_stop(val_loss):
            print("Early stopping")
            break

    val_auc = training_utils.test(model, val_set)
    curve_data = [train_losses, val_losses, train_scores, val_scores]

    return val_auc, model, curve_data

def run_multiple_seeds(datasets,experiment_params,negative_sampler):
    experiment_metrics = []
    models = []
    curves = []
    for seed_dataset in datasets:
        train_data, val_data = seed_dataset
        seed_auc, trained_model, training_curve = run_experiment(experiment_params,train_data,val_data,negative_sampler)
        experiment_metrics.append(seed_auc)
        models.append(trained_model)
        curves.append(training_curve)
    
    metrics = (np.mean(experiment_metrics),np.std(experiment_metrics))
    
    return [metrics, models, curves]
#%%
seeds = [4,5,6,7,8]
data = []
for seed in seeds:
    datasets, node_map = training_utils.load_data(data_folder+f"seed_{seed}/")
    data.append(datasets)

full_set = torch.load(data_folder+f"seed_{seeds[-1]}/full_dataset.pt")
training_params = {"weight_decay":0.001, "lr":0.001, "epochs":400, "patience":10, "delta":0.1, "feature_type":"lsa_scaled","feature_dim":32}

negative_sampler = training_utils.NegativeSampler(full_set,("gene_protein","gda","disease"),full_set["gene_protein"]["degree_gda"],full_set["disease"]["degree_gda"])
auc, models, curves = run_multiple_seeds(data,training_params,negative_sampler)

with open(f'{models_folder}validation_auc.txt', 'w') as f:
    f.write(f"mean AUROC:{round(auc[0],3)} +- {round(auc[1],3)}. /n Num seeds: {len(data)}")

with open(f"{models_folder}training_parameters.pickle", 'wb') as handle:
    pickle.dump(training_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

for model, curve, seed in zip(models,curves,seeds):
    folder = models_folder+"seeds/"
    model_name = f"final_model_{seed}"
    curve_data = np.array(curves, dtype=object)
    np.save(folder+model_name+"_curve.npy", curve_data) # curve_data shape = [train_losses, val_losses, train_scores, val_scores]
    torch.save(model.state_dict(), folder+model_name+".pth")