#%%
import pandas as pd
#%%
graph_data = "../../data/processed/graph_data_nohubs/"
reports_tfidf = "../../reports/reports_nohubs/analisis_tfidf/"
reports_lsa = "../../reports/reports_nohubs/analisis_lsa/"
louvain_analysis = pd.read_pickle(graph_data+"louvain_top_terms.pkl")
infomap_analysis = pd.read_pickle(graph_data+"infomap_top_terms.pkl")
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
    
    monogram_match =  df[df.top_5_monogram.apply(lambda x: keyword in x)]
    monogram_match["score"] = monogram_match.apply(lambda x: match_keyword_to_score(x.top_5_monogram, x.top_5_monogram_score, keyword), axis=1)
    monogram_match = monogram_match.sort_values(by="score", ascending=False)

    bigram_match = df[df.top_5_bigram.apply(lambda x: any(keyword in s for s in x))]
    bigram_match["score"] = bigram_match.apply(lambda x: match_keyword_to_score(x.top_5_bigram, x.top_5_bigram_score, keyword), axis=1)
    bigram_match = bigram_match.sort_values(by="score", ascending=False)

    trigram_match = df[df.top_5_trigram.apply(lambda x: any(keyword in s for s in x))]
    trigram_match["score"] = trigram_match.apply(lambda x: match_keyword_to_score(x.top_5_trigram, x.top_5_trigram_score, keyword), axis=1)
    trigram_match = trigram_match.sort_values(by="score", ascending=False)

    result = pd.concat([monogram_match,bigram_match,trigram_match]).drop_duplicates(subset=["comunidad"], keep="first")
    result.drop(result[result.score==0].index, inplace=True)

    return result
#%%
find_cluster("infomap","thrombocytopenia")