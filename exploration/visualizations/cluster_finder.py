#%%
import pandas as pd
import numpy as np
#%%
reports_tfidf = "../../reports/reports_nohubs/analisis_tfidf/"
louvain_analysis = pd.read_pickle(reports_tfidf+"louvain_analysis_checkpoint.pkl")
infomap_analysis = pd.read_pickle(reports_tfidf+"infomap_analysis_checkpoint.pkl")
#%%

def match_keyword_to_score(word_col,score_col,keyword):
    mask = [keyword in s for s in word_col]
    score = score_col[mask].max()
    return score

def find_cluster(partition, keyword:str):
    if partition == "infomap":
        df = infomap_analysis
    elif partition == "louvain":
        df = louvain_analysis
    else: 
        print("Not a valid partition")
    
    monogram_match =  df[df.top_5_monograms.apply(lambda x: keyword in x)]
    monogram_match["score"] = monogram_match.apply(lambda x: match_keyword_to_score(x.top_5_monograms, x.top_5_monograms_score, keyword), axis=1)
    monogram_match = monogram_match.sort_values(by="score", ascending=False)

    bigram_match = df[df.top_5_bigrams.apply(lambda x: any(keyword in s for s in x))]
    bigram_match["score"] = bigram_match.apply(lambda x: match_keyword_to_score(x.top_5_bigrams, x.top_5_bigrams_score, keyword), axis=1)
    bigram_match = bigram_match.sort_values(by="score", ascending=False)

    trigram_match = df[df.top_5_trigrams.apply(lambda x: any(keyword in s for s in x))]
    trigram_match["score"] = trigram_match.apply(lambda x: match_keyword_to_score(x.top_5_trigrams, x.top_5_trigrams_score, keyword), axis=1)
    trigram_match = trigram_match.sort_values(by="score", ascending=False)

    result = pd.concat([monogram_match,bigram_match,trigram_match]).drop_duplicates(subset=["comunidad"], keep="first")
    result.drop(result[result.score==0].index, inplace=True)

    return result
#%%
find_cluster("louvain","carcinoma")
#%%
data_processed = "../../data/processed/"
data_external = "../../data/external/"
graph_data = data_processed + "graph_data_nohubs/"

graph_node_data = pd.read_csv(graph_data+"nohub_graph_node_data.csv")
graph_edge_data = pd.read_csv(graph_data+"nohub_graph_edge_data.csv").rename(columns={"relation":"edge_type"})

retinitis = graph_node_data[graph_node_data.comunidades_louvain == 54]
joubert = graph_node_data[graph_node_data.comunidades_louvain == 20]
#%%
disease_attributes = pd.read_csv(graph_data+"nohub_disease_attributes.csv")
orig_disgenet = pd.read_csv(data_external+"disease_mappings_to_attributes.tsv",sep="\t")
curated_edges = pd.read_csv(data_external+"curated_gene_disease_associations.tsv", sep="\t")
#%%
retinitis_ejemplos = [29377, 20693, 27388, 30769]
retinitis_ejemplos_ids = disease_attributes.set_index("node_index").loc[retinitis_ejemplos, "node_id"].values

graph_edge_data.set_index("x_index").loc[retinitis_ejemplos]
orig_disgenet.set_index("diseaseId").loc[retinitis_ejemplos_ids]
curated_edges.set_index("diseaseId").loc[retinitis_ejemplos_ids]