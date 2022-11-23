#%%
import pandas as pd
#%%
graph_data = "../../data/processed/graph_data_nohubs/"
louvain_analysis = pd.read_pickle(graph_data+"louvain_top_terms.pkl")
infomap_analysis = pd.read_pickle(graph_data+"infomap_top_terms.pkl")
#%%

def match_keyword_to_score(word_col,score_col,keyword):
    mask = [keyword in s for s in word_col]
    score = score_col[mask].max()
    return score

def get_match_rows(df,col,keyword):
    match =  df[df[col].apply(lambda x: keyword in x)].copy()
    if len(match) != 0:
        match["score"] = match.apply(lambda x: match_keyword_to_score(x[col], x[f"{col}_score"], keyword), axis=1)
        match = match.sort_values(by="score", ascending=False)
    return match

def find_cluster(partition, keyword:str):
    if partition == "infomap":
        df = infomap_analysis
    elif partition == "louvain":
        df = louvain_analysis
    else: 
        print("Not a valid partition")

    monogram_match =  get_match_rows(df,"top_5_monogram",keyword)
    bigram_match = get_match_rows(df,"top_5_bigram",keyword)
    trigram_match = get_match_rows(df,"top_5_trigram",keyword)

    result = pd.concat([monogram_match,bigram_match,trigram_match]).drop_duplicates(subset=["comunidad"], keep="first")
    result.drop(result[result.score==0].index, inplace=True)

    return result
#%%
find_cluster("infomap","mosquito")
#%%
find_cluster("louvain","mosquito")

