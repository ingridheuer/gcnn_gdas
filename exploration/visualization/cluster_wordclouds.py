#%%
import pandas as pd
import matplotlib.pyplot as plt
# import plotly.express as px
from wordcloud import WordCloud
#%%
def load_dtm_matrices(path,name,N):
    term_path = f"{path}top_{N}_terms_{name}.csv"
    score_path = f"{path}top_{N}_score_{name}.csv"
    term_matrix = pd.read_csv(term_path, index_col=0)
    score_matrix = pd.read_csv(score_path, index_col=0)
    return term_matrix, score_matrix

def plot_wordcloud(partition,cluster,cmap,background_color="white"):
    if partition == "infomap":
        terms,scores = load_dtm_matrices("../../reports/summary/","infomap",100)
    elif partition == "louvain":
        terms,scores = load_dtm_matrices("../../reports/summary/","louvain",100)
    else:
        print("Not a valid partition")

    cluster_terms = terms.loc[cluster]
    cluster_scores = scores.loc[cluster]
    frequencies = dict(zip(cluster_terms,cluster_scores))
    wordcloud = WordCloud(colormap=cmap,background_color=background_color).generate_from_frequencies(frequencies=frequencies)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off") 
    plt.show()

# def plot_term_distribution(partition,cluster):
#     if partition == "infomap":
#         terms,scores = load_dtm_matrices("../../reports/summary/","infomap",100)
#     elif partition == "louvain":
#         terms,scores = load_dtm_matrices("../../reports/summary/","louvain",100)
#     else:
#         print("Not a valid partition")
    
#     cluster_terms = terms.loc[cluster]
#     cluster_scores = scores.loc[cluster]
#     frequencies = dict(zip(cluster_terms,cluster_scores))

#     fig = px.bar(frequencies,width=800, height=400, title="TF-IDF monogram distribution").update_layout(yaxis_title="Monograms",xaxis_title="TF-IDF value")
#     fig.show()

#%%
plot_wordcloud("infomap",525,None,"white")
#%%
plot_wordcloud("louvain",31,None,"white")
#%%