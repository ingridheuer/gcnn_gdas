#%%
import pandas as pd
#%%
data_processed = "../../data/processed/"
data_interim = "../../data/interim/"
data_external = "../../data/external/"
graph_data = data_processed + "graph_data_nohubs/"
entropy_reports = "../../reports/reports_nohubs/analisis_tfidf/entropy/"
lsa_reports = "../../reports/reports_nohubs/analisis_lsa/"
#%%

def make_summary(particion:str):
    meansim = pd.read_csv(lsa_reports+f"{particion}_meansim.csv")
    entropy = pd.read_csv(entropy_reports+f"entropy_{particion}.csv").drop(columns=["comunidad","tamaño"])
    keywords = pd.read_pickle(graph_data+f"{particion}_top_terms.pkl").drop(columns="comunidad")

    summary = pd.concat([meansim,entropy,keywords], axis=1)
    return summary
#%%
infomap_summary = make_summary("infomap")
louvain_summary = make_summary("louvain")
#%%
infomap_summary.to_pickle("../../reports/reports_nohubs/infomap_summary.pkl")
louvain_summary.to_pickle("../../reports/reports_nohubs/louvain_summary.pkl")