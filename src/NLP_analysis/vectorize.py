#%%
import pandas as pd
import numpy as np
import random
import pickle
from scipy import sparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import cosine_similarity

import config
#%%
seed = 16
random.seed(seed)
np.random.seed(seed)
#%%
data_processed = "../../data/processed/"
data_interim = "../../data/interim/"
data_external = "../../data/external/"
graph_data = data_processed + "graph_data_nohubs/"
#%%
#Cargo corpus preprocesado
with open(graph_data+"processed_node_documents.pickle", 'rb') as handle:
    processed_node_documents = pickle.load(handle)

args = config.nlp_args["TFIDF"]

stop_words = text.ENGLISH_STOP_WORDS.union(args["custom_stopwords"])
#%%
# Dataframes
graph_node_data = pd.read_csv(graph_data+"nohub_graph_node_data.csv")

tamaños_infomap = graph_node_data.comunidades_infomap.dropna().value_counts()
tamaños_louvain = graph_node_data.comunidades_louvain.dropna().value_counts()

infomap_clusters = pd.DataFrame(tamaños_infomap).reset_index().rename(columns={"index":"comunidad","comunidades_infomap":"tamaño"}).astype({"comunidad":"int"})
louvain_clusters = pd.DataFrame(tamaños_louvain).reset_index().rename(columns={"index":"comunidad","comunidades_louvain":"tamaño"}).astype({"comunidad":"int"})
#%%
#Functions
def get_tfidf_df(vectorizer,corpus,ids,df=True):
    X = vectorizer.fit_transform(corpus)
    if df:
        X = pd.DataFrame.sparse.from_spmatrix(X, index=ids, columns=vectorizer.get_feature_names_out())
    return X

def tfidf_similarity(vectorizer,corpus,ids,df=True):
    scores = get_tfidf_df(vectorizer,corpus,ids,False)
    similarity_matrix = cosine_similarity(scores,scores,False)
    if df:
        similarity_matrix = pd.DataFrame.sparse.from_spmatrix(similarity_matrix, index=ids, columns=ids)
    return similarity_matrix

def cluster_as_document(cluster_id,cluster_algorithm):

    cluster_nodes = graph_node_data.loc[graph_node_data[cluster_algorithm] == cluster_id, "node_index"].values
    cluster_corpus = [processed_node_documents[node_index] for node_index in cluster_nodes]
    cluster_document = " ".join(cluster_corpus)

    return cluster_document
#%%
vectorizers = [TfidfVectorizer(tokenizer = None , stop_words=stop_words, ngram_range=ngram_range, min_df=args["min_df"],max_df=args["max_df"],max_features=args["max_features"])for ngram_range in [(1,1),(2,2),(3,3),(1,3)]]
#%%
# NODES-AS-DOCUMENTS
# Get TF-IDF matrices and save in sparse format

corpus = [*processed_node_documents.values()]
ids = [*processed_node_documents.keys()]

document_term_matrix = []
document_similarity_matrix = []

print("Vectorizing nodes ...")
for i,vectorizer in enumerate(vectorizers):
    document_term_matrix.append(get_tfidf_df(vectorizer, corpus,ids))
    document_similarity_matrix.append(tfidf_similarity(vectorizer,corpus,ids))


path = graph_data+"tfidf_nodos/"

for i,mat in enumerate(document_term_matrix):
    cols = [s.replace(" ", "_") for s in mat.columns.values]
    rows = mat.index.values
    vals = mat.sparse.to_coo()
    sparse.save_npz(f"{path}matriz_nodos_tfidf_{i}.npz",vals)
    np.savetxt(f"{path}rows_tfidf_nodos_{i}.txt", rows)
    np.savetxt(f"{path}cols_tfidf_nodos_{i}.txt", cols,fmt="%s")

print(f"Vectorizing nodes done. Data saved at {path}")
#%%

# CLUSTERS-AS-DOCUMENTS
# Get TF-IDF matrices and save in sparse format

infomap_ids = infomap_clusters.comunidad.values
louvain_ids = louvain_clusters.comunidad.values

corpus_infomap = [cluster_as_document(cluster_id,"comunidades_infomap") for cluster_id in infomap_ids]
corpus_louvain = [cluster_as_document(cluster_id,"comunidades_louvain") for cluster_id in louvain_ids]

tfidf_infomap = []
tfidf_louvain = []

similarity_infomap = []
similarity_louvain = []

print("Vectorizing clusters ...")
for i,vectorizer in enumerate(vectorizers):
    tfidf_infomap.append(get_tfidf_df(vectorizer, corpus_infomap,infomap_ids))
    tfidf_louvain.append(get_tfidf_df(vectorizer,corpus_louvain,louvain_ids))

    similarity_infomap.append(tfidf_similarity(vectorizer, corpus_infomap, infomap_ids))
    similarity_louvain.append(tfidf_similarity(vectorizer, corpus_louvain, louvain_ids))

path_infomap = graph_data + "tfidf_infomap/"
path_louvain = graph_data + "tfidf_louvain/"

for i,mat in enumerate(tfidf_infomap):
    cols = [s.replace(" ", "_") for s in mat.columns.values]
    rows = mat.index.values
    vals = mat.sparse.to_coo()
    sparse.save_npz(f"{path_infomap}matriz_tfidf_infomap_{i}.npz",vals)
    np.savetxt(f"{path_infomap}rows_tfidf_infomap_{i}.txt", rows)
    np.savetxt(f"{path_infomap}cols_tfidf_infomap_{i}.txt", cols,fmt="%s")

for i,mat in enumerate(tfidf_louvain):
    cols = [s.replace(" ", "_") for s in mat.columns.values]
    rows = mat.index.values
    vals = mat.sparse.to_coo()
    sparse.save_npz(f"{path_louvain}matriz_tfidf_louvain_{i}.npz",vals)
    np.savetxt(f"{path_louvain}rows_tfidf_louvain_{i}.txt", rows)
    np.savetxt(f"{path_louvain}cols_tfidf_louvain_{i}.txt", cols,fmt="%s")

print(f"Vectorizing clusters done. Data saved at {path_infomap} , {path_louvain}")