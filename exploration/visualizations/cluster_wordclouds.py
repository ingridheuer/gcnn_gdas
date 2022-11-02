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

def plot_wordcloud(cluster_term_matrix,cluster,num_terms,cmap,background_color="white"):
    cluster_series = cluster_term_matrix.loc[cluster].sort_values(ascending=False)[0:num_terms]
    wordcloud = WordCloud(colormap=cmap,background_color=background_color).generate_from_frequencies(frequencies=cluster_series)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()

def plot_term_distribution(cluster,cluster_term_matrix):
    fig = px.bar(cluster_term_matrix.loc[cluster].sort_values(ascending=False)[0:10],width=800, height=400, title="TF-IDF monogram distribution").update_layout(yaxis_title="Monograms",xaxis_title="TF-IDF value")
    fig.show()
#%%
louvain_path = graph_data + "tfidf_louvain/"
louvain_term_matrix = load_sparse_dataframe(louvain_path+"matriz_tfidf_louvain_0.npz", louvain_path+"rows_tfidf_louvain_0.txt",louvain_path+"cols_tfidf_louvain_0.txt")

cluster_analysis = pd.read_pickle(reports_tfidf+"louvain_analysis_checkpoint.pkl")

dense_dat = louvain_term_matrix.sparse.to_dense()
#%%
plot_wordcloud(dense_dat,3,70,None,"white")
#%%
plot_wordcloud(dense_dat,31,70,None,"white")
#%%