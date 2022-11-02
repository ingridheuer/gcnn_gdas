#%%
import pandas as pd
import numpy as np
#%%
def match_keyword_to_score(word_col,score_col,keyword):
    mask = [keyword in s for s in word_col]
    score = score_col[mask].max()
    return score

def find_cluster(df, keyword:str):
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
reports_tfidf = "../../reports/reports_nohubs/analisis_tfidf/"
louvain_analysis = pd.read_pickle(reports_tfidf+"louvain_analysis_checkpoint.pkl")
infomap_analysis = pd.read_pickle(reports_tfidf+"infomap_analysis_checkpoint.pkl")
#%%
find_cluster(infomap_analysis,"breast")
# %%
