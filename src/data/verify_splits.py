#%%
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_venn import venn3

import sys
sys.path.append("..")
from models import prediction_utils, training_utils
#%%
data_folder = "../../data/processed/graph_data_nohubs/merged_types/split_dataset/"
models_folder = "../../models/final_model/"
feature_folder = "../../data/processed/feature_data/"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pred_edge_type = ("gene_protein","gda","disease")
seeds = [4,5,6,7,8]

#load data
data = []
for seed in seeds:
    datasets, node_map = training_utils.load_data(data_folder+f"seed_{seed}/",load_test=True)
    data.append(datasets)
#%%
def get_edge_sets(heterodata, node_map=node_map):
    mapped = prediction_utils.MappedDataset(heterodata, node_map,("gene_protein", "gda", "disease"))
    mapped_df = mapped.dataframe

    mapped_df["edges"] = mapped_df[["gene_protein_source","disease_target"]].values.tolist()
    mapped_df["edges"] = mapped_df.edges.apply(lambda x: tuple(x))
    supervision_edges = set(mapped_df[(mapped_df.edge_type == "supervision") & (mapped_df.label == 1)].edges.values)
    propagation_edges = set(mapped_df[mapped_df.edge_type == "message_passing"].edges.values)

    return mapped_df,propagation_edges,supervision_edges

mapped_sets = []
supervision_sets = []
propagation_sets = []

# format = [[train_sup, val_sup, test_sup],....] for each seed
for seed in data:
    train,val,test = seed
    seed_supervision = []
    seed_propagation = []
    for split in seed:
        mapped_df,propagation, supervision = get_edge_sets(split)
        seed_supervision.append(supervision)
        seed_propagation.append(propagation)
    
    mapped_sets.append(mapped_df)
    supervision_sets.append(seed_supervision)
    propagation_sets.append(seed_propagation)
#%%
# verifico leaks entre enlaces de supervision de mismo seed
# se generan leaks negativos entre val y test, del orden del 0.05% de los enlaces totales
# esto es esperable por como se genera la muestra, la contribución no es significativa
train_val_leak = []
train_test_leak = []
val_test_leak = []

check_negatives = []
for i,seed in enumerate(supervision_sets):
    train_val_leak.append(seed[0]&seed[1])
    train_test_leak.append(seed[0]&seed[1])

    vt_leak = seed[1]&seed[2]
    val_test_leak.append(len(vt_leak))

    loc_edges = list(vt_leak)
    df = mapped_sets[i]
    num_positive = df.set_index("edges").loc[loc_edges].label.values.sum()
    check_negatives.append(num_positive)

results = {"Compartidos":val_test_leak, "Número de enlaces positivos":check_negatives}
results_df = pd.DataFrame(results)
results_df["Compartidos %"] = round((results_df["Compartidos"]*100)/len(supervision_sets[0][1]),2)
results_df["Total enlaces de supervisión en test"] = [len(x[2]) for x in supervision_sets]
results_df["Total enlaces de supervisión en val"] = [len(x[1]) for x in supervision_sets]
# results_df.to_csv("../../reports/verify_splits/vt_leak_check.csv")

#%%
aver = list(val_test_leak[0])
avor = mapped_sets[0]
avor.set_index("edges").loc[aver].label.values.sum()
#%%
#verifico que los conjuntos de propagación "crecen" de train a test
#esto se cumple y además todos crecen el mismo número
train_val_prop = []
train_test_prop = []
val_test_prop = []
for seed in propagation_sets:
    train_val_prop.append(len(seed[0]&seed[1]))
    train_test_prop.append(len(seed[0]&seed[1]))
    val_test_prop.append(len(seed[1]&seed[2]))

#%%
#lo ploteo para uno solo pero pasa lo mismo para los tres
v = venn3(subsets=propagation_sets[0],set_labels=["train","val","test"])
v.get_label_by_id("A").set_y(0.0)
v.get_label_by_id("A").set_x(-0.3)
v.get_label_by_id("111").set_x(-0.35)
v.get_label_by_id("111").set_y(-0.05)
v.get_label_by_id("B").set_y(0.35)
v.get_label_by_id("C").set_x(0.5)
v.get_label_by_id("C").set_y(-0.25)
v.get_label_by_id("001").set_y(-0.2)
v.get_label_by_id("011").set_y(0.3)
v.get_label_by_id("011").set_x(0.4)

plt.title("Enlaces de propagación")
plt.show()
#%%
#verifico que los enlaces de supervisión entre diferentes splits no sean iguales
sup_sets = []

for k in range(3):
    sub_sup = []
    for i in range(5):
        rows = []
        for j in range(5):
            set_idx = k
            i_set = supervision_sets[i][set_idx]
            j_set = supervision_sets[j][set_idx]
            intersection = len(i_set&j_set)
            union = len(i_set|j_set)
            rows.append(round((intersection*100)/len(i_set),2))
        sub_sup.append(rows)
    sup_sets.append(sub_sup)
#%%
names = ["train","validation","test"]
for k,split in enumerate(sup_sets):
    matrix = np.array(split)
    fig, ax = plt.subplots()
    im = ax.imshow(matrix)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(split)), labels=[4,5,6,7,8])
    ax.set_yticks(np.arange(len(split)), labels=[4,5,6,7,8])

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(split)):
        for j in range(len(split)):
            text = ax.text(j, i, matrix[i, j],
                        ha="center", va="center", color="w")

    ax.set_title(f"% Enlaces de supervisión compartidos entre semillas \n {names[k]} set")
    fig.tight_layout()
    plt.show()
#%%
prop_sets = []

for k in range(3):
    sub_prop = []
    for i in range(5):
        rows = []
        for j in range(5):
            set_idx = k
            i_set = propagation_sets[i][set_idx]
            j_set = propagation_sets[j][set_idx]
            intersection = len(i_set&j_set)
            union = len(i_set|j_set)
            rows.append(round((intersection*100)/len(i_set),2))
        sub_prop.append(rows)
    prop_sets.append(sub_prop)
#%%
names = ["train","validation","test"]
for k,split in enumerate(prop_sets):
    matrix = np.array(split)
    fig, ax = plt.subplots()
    im = ax.imshow(matrix)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(split)), labels=[4,5,6,7,8])
    ax.set_yticks(np.arange(len(split)), labels=[4,5,6,7,8])

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(split)):
        for j in range(len(split)):
            text = ax.text(j, i, matrix[i, j],
                        ha="center", va="center", color="w")

    ax.set_title(f"% Enlaces de propagación compartidos entre semillas \n {names[k]} set")
    fig.tight_layout()
    plt.show()
#%%