#%%
import torch
import pandas as pd
import numpy as np
from torch_geometric import seed_everything
import final_model, training_utils, prediction_utils
import pickle
from tqdm import tqdm
seed = 4

seed_everything(seed)
#%%
data_folder = "../../data/processed/graph_data_nohubs/merged_types/split_dataset/"
models_folder = "../../models/final_model/"
feature_folder = "../../data/processed/feature_data/"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#%%
#load data
dataset, node_map = training_utils.load_data(data_folder+f"seed_{seed}/",load_test=True)
test_data = dataset[2]
node_df = pd.read_csv(data_folder+f"seed_{seed}/tensor_df.csv",index_col=0).set_index("node_index",drop=True)

with open(f"{models_folder}training_parameters.pickle", 'rb') as handle:
    params = pickle.load(handle)

#initialize features in test data
test_data  = training_utils.initialize_features(test_data,params["feature_type"],params["feature_dim"],feature_folder)

#load model
weights_path = models_folder+f"seeds/final_model_{seed}.pth"
weights = torch.load(weights_path)
model = final_model.Model(test_data.metadata(),[("gene_protein","gda","disease")])
model.load_state_dict(weights)
#%%
encodings_dict = training_utils.get_encodings(model,test_data)
mapped_dataset = prediction_utils.MappedDataset(test_data,node_map,("gene_protein","gda","disease"))
mapped_df = mapped_dataset.dataframe
predictor = prediction_utils.Predictor(node_df,encodings_dict)
#%%
diseases = node_df[node_df.node_type == "disease"].index.values
genes = node_df[node_df.node_type == "gene_protein"].index.values
#%%
num_pred = 50

#diseases
remove_edges = mapped_df[(mapped_df.edge_type == "message_passing")| (mapped_df.label == 1)][["gene_protein","disease"]]
# disease_rankings = []
# # disease_scores = []
# for disease in tqdm(diseases):
#     pred = predictor.prioritize_one_vs_all(disease)
#     if disease in remove_edges.disease.values:
#         to_remove = np.array(mapped_dataset.dataframe.set_index("disease").loc[disease].gene_protein)
#         pred = pred[~pred.node_index.apply(lambda x: x in to_remove)].reset_index(drop=True)
#     ranking = pred[:50]["node_index"].values
#     # scores = pred[:50]["score"].values

#     disease_rankings.append(pred)

# disease_predictions = pd.DataFrame(disease_rankings)
# # disease_scores = pd.DataFrame(disease_scores)
# disease_predictions.to_csv("../../reports/model_predictions/disease_predictions.csv")
# # disease_scores.to_csv("../../reports/model_predictions/disease_scores.csv")
#%%
#genes
gene_rankings = []
# gene_scores = []
for gene in tqdm(genes):
    pred = predictor.prioritize_one_vs_all(gene)
    if gene in remove_edges.gene_protein.values:
        to_remove = np.array(mapped_dataset.dataframe.set_index("gene_protein").loc[gene].disease)
        pred = pred[~pred.node_index.apply(lambda x: x in to_remove)].reset_index(drop=True)
    ranking = pred[:50]["node_index"].values
    # scores = pred[:50]["score"].values

    gene_rankings.append(pred)

gene_predictions = pd.DataFrame(gene_rankings)
# gene_scores = pd.DataFrame(gene_scores)
gene_predictions.to_csv("../../reports/model_predictions/gene_predictions.csv")
# gene_predictions.to_csv("../../reports/model_predictions/gene_scores.csv")
# %%
