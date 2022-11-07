#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from scipy import sparse
from wordcloud import WordCloud
#%%
data_processed = "../../data/processed/"
data_interim = "../../data/interim/"
data_external = "../../data/external/"
graph_data = data_processed + "graph_data_nohubs/"
reports_tfidf = "../../reports/reports_nohubs/analisis_tfidf/"
#%%
def load_sparse_dataframe(matrix_path,row_path,column_path,cols_str=True):
    mat = sparse.load_npz(matrix_path)
    row = np.loadtxt(row_path)
    if cols_str:
        col = np.loadtxt(column_path, dtype="str")
    else:
        col = np.loadtxt(column_path)
        
    df = pd.DataFrame.sparse.from_spmatrix(mat, index=row, columns=col)
    return df

#%%
louvain_path = graph_data + "tfidf_louvain/"
infomap_path = graph_data + "tfidf_infomap/"

louvain_term_matrix = load_sparse_dataframe(louvain_path+"matriz_tfidf_louvain_0.npz", louvain_path+"rows_tfidf_louvain_0.txt",louvain_path+"cols_tfidf_louvain_0.txt")
infomap_term_matrix = load_sparse_dataframe(infomap_path+"matriz_tfidf_infomap_0.npz", infomap_path+"rows_tfidf_infomap_0.txt",infomap_path+"cols_tfidf_infomap_0.txt")

dense_dat_louvain = louvain_term_matrix.sparse.to_dense()
dense_dat_infomap = infomap_term_matrix.sparse.to_dense()
#%%
def plot_wordcloud(partition,cluster,num_terms,cmap,background_color="white"):
    if partition == "infomap":
        cluster_term_matrix = dense_dat_infomap
    elif partition == "louvain":
        cluster_term_matrix = dense_dat_louvain
    else:
        print("Not a valid partition")

    cluster_series = cluster_term_matrix.loc[cluster].sort_values(ascending=False)[0:num_terms]
    wordcloud = WordCloud(colormap=cmap,background_color=background_color).generate_from_frequencies(frequencies=cluster_series)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()

def plot_term_distribution(partition,cluster):

    if partition == "infomap":
        cluster_term_matrix = dense_dat_infomap
    elif partition == "louvain":
        cluster_term_matrix = dense_dat_louvain
    else:
        print("Not a valid partition")

    fig = px.bar(cluster_term_matrix.loc[cluster].sort_values(ascending=False)[0:10],width=800, height=400, title="TF-IDF monogram distribution").update_layout(yaxis_title="Monograms",xaxis_title="TF-IDF value")
    fig.show()

#%%
plot_wordcloud("louvain",3,70,None,"white")
#%%
plot_wordcloud("louvain",31,70,None,"white")
#%%