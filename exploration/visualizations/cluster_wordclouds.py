#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from matplotlib_venn import venn3, venn2
import random
import plotly.io as pio
from scipy import sparse
import os 
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
#%%
data_processed = "../../data/processed/"
data_interim = "../../data/interim/"
data_external = "../../data/external/"

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
louvain_term_matrix = load_sparse_dataframe(data_processed+"tfidf_louvain/matriz_tfidf_louvain_0.npz", data_processed+"tfidf_louvain/rows_tfidf_louvain_0.txt",data_processed+"tfidf_louvain/cols_tfidf_louvain_0.txt")

cluster_analysis = pd.read_pickle("../../reports/tfidf/louvain_analysis_checkpoint.pkl")

dense_dat = louvain_term_matrix.sparse.to_dense()
#%%
plot_wordcloud(dense_dat,2,70,None,"white")
#%%
plot_wordcloud(dense_dat,10,70,None,"white")