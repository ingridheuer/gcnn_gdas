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
seeds = [4,5,6,7,8]

#load data
data = []
for seed in seeds:
    datasets, node_map = training_utils.load_data(data_folder+f"seed_{seed}/",load_test=True)
    data.append(datasets)

with open(f"{models_folder}training_parameters.pickle", 'rb') as handle:
    params = pickle.load(handle)

#initialize features in test data
test_data = []
for dataset in data:
    test_set = dataset[2]
    test_data.append(training_utils.initialize_features(test_set,params["feature_type"],params["feature_dim"],feature_folder))

#load models
models = []
for seed in seeds:
    weights_path = models_folder+f"seeds/final_model_{seed}.pth"
    weights = torch.load(weights_path)
    model = final_model.Model(test_data[-1].metadata(),[("gene_protein","gda","disease")])
    model.load_state_dict(weights)
    models.append(model)

#test models
aucs = []
for model,test_dataset in zip(models,test_data):
    auc = training_utils.test(model,test_dataset)
    aucs.append(auc)

test_auc, test_std = np.mean(aucs), np.std(aucs)

with open(f'{models_folder}test_auc.txt', 'w') as f:
    f.write(f"mean test AUROC:{round(test_auc,3)} +- {round(test_std,3)}. /n Num seeds: {len(data)}")