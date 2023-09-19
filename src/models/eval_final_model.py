#%%
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, RocCurveDisplay, accuracy_score, average_precision_score, precision_score,recall_score
import matplotlib.pyplot as plt
from torch_geometric import seed_everything
import final_model, training_utils, prediction_utils
import pickle
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
predictor = prediction_utils.Predictor(node_df,encodings_dict)
#%%
preds = predictor.predict_supervision_edges(test_data,("gene_protein","gda","disease"))
#%%

y_true = preds.label.values
y_score = preds.score.values.round(3)

RocCurveDisplay.from_predictions(y_true,y_score)
fpr,tpr,thresholds = roc_curve(y_true,y_score)
#%%
th_optimal = thresholds[np.argmax(tpr - fpr)]
optimal_arg = np.argwhere(thresholds == th_optimal)
#%%
plt.figure()
plt.plot(fpr,tpr, label="LSA-SAGE")
plt.plot(np.linspace(0,1,len(fpr)),np.linspace(0,1,len(fpr)), "--", label="Modelo aleatorio")
plt.scatter(fpr[optimal_arg],tpr[optimal_arg], label=f"Umbral de clasificación ideal = {th_optimal:.2f}")
plt.legend()
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic curve")
#%%
#%%
acc_score = round(accuracy_score(y_true,y_score.round()),2)
ap_score = round(average_precision_score(y_true,y_score),2)
#%%