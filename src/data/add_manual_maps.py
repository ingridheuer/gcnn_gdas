#%%
import pandas as pd

data_processed = "../../data/processed/"
data_interim = "../../data/interim/"
data_external = "../../data/external/"

disease_mappings = pd.read_csv(data_interim+"disease_vocab_mapping.csv")
manual_maps = pd.read_csv(data_interim+"manual_disease_maps.csv")
prime_grouped_bert = pd.read_csv(data_external + "kg_grouped_diseases_bert_map.csv")

manual_maps = pd.merge(manual_maps[["diseaseId","code","name","node_index","vocabularyName"]], prime_grouped_bert[["node_id","group_id_bert","group_name_bert","group_name_auto"]],left_on="code",right_on="node_id",how="left").drop(columns="node_id")
disease_mappings = disease_mappings.set_index("diseaseId").drop(manual_maps.diseaseId.values).reset_index()
disease_mappings = pd.concat([disease_mappings,manual_maps]).rename(columns={"node_index":"prime_node_index","code":"mondo_id"})

disease_mappings.to_csv(data_processed+"graph_data/disease_mappings.csv")
# %%
