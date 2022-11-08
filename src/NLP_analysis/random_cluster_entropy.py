#%%
import numpy as np
import pandas as pd
import random
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text

import nlp_utils
from config import nlp_args
#%%
seed = 16
random.seed(seed)
np.random.seed(seed)
#%%
data_processed = "../../data/processed/"
data_interim = "../../data/interim/"
data_external = "../../data/external/"
graph_data = data_processed + "graph_data_nohubs/"
path_infomap = graph_data + "tfidf_infomap/"
path_louvain = graph_data + "tfidf_louvain/"
entropy_reports = "../../reports/reports_nohubs/analisis_tfidf/entropy/random_entropy_data/"

graph_node_data = pd.read_csv(graph_data+"nohub_graph_node_data.csv")

infomap_clusters, louvain_clusters = nlp_utils.get_cluster_dataframes(graph_node_data)
infomap_ids = infomap_clusters.comunidad.values
louvain_ids = louvain_clusters.comunidad.values

#Cargo corpus preprocesado
with open(graph_data+"processed_node_documents.pickle", 'rb') as handle:
    processed_node_documents = pickle.load(handle)

#%%
def cluster_as_document(cluster_nodes):
    cluster_corpus = [processed_node_documents[node_index] for node_index in cluster_nodes]
    cluster_document = " ".join(cluster_corpus)

    return cluster_document

def shuffle_clusters(node_column, cluster_columns):
    shuffled_nodes = node_column.sample(frac=1).reset_index(drop=True)
    new_clusters = pd.concat([shuffled_nodes,cluster_columns], axis=1)

    infomap_shuffled = new_clusters.groupby("comunidades_infomap")["node_index"].apply(list)
    louvain_shuffled = new_clusters.groupby("comunidades_louvain")["node_index"].apply(list)
    return infomap_shuffled, louvain_shuffled

def build_random_corpus(node_column,cluster_columns):
    print("Building corpus ...")

    infomap_shuffled, louvain_shuffled = shuffle_clusters(node_column,cluster_columns)

    corpus_infomap = [cluster_as_document(infomap_shuffled[cluster]) for cluster in infomap_ids]
    corpus_louvain = [cluster_as_document(louvain_shuffled[cluster]) for cluster in louvain_ids]

    return corpus_infomap, corpus_louvain

def get_tfidf_df(vectorizer,corpus,ids,df=True):
    X = vectorizer.fit_transform(corpus)
    if df:
        X = pd.DataFrame.sparse.from_spmatrix(X, index=ids, columns=vectorizer.get_feature_names_out())
    return X

def vectorize(corpus_infomap, corpus_louvain, vectorizers):
    tfidf_infomap = []
    tfidf_louvain = []

    print("Vectorizing clusters ...")
    for i,vectorizer in enumerate(vectorizers):
        tfidf_infomap.append(get_tfidf_df(vectorizer, corpus_infomap,infomap_ids).loc[infomap_random_ids])
        tfidf_louvain.append(get_tfidf_df(vectorizer,corpus_louvain,louvain_ids).loc[louvain_random_ids])
    
    return tfidf_infomap, tfidf_louvain

def cluster_entropy(tfidf_infomap, tfidf_louvain):
    print("Computing entropy ...")
    entropias_infomap = []
    for i, mat in enumerate(tfidf_infomap):
        name = "entropia_"+str(i)
        entropia_series = mat.apply(lambda x: nlp_utils.get_entropy(x), axis=1, raw=True).rename(name)
        entropias_infomap.append(entropia_series)

    entropias_louvain = []
    for i, mat in enumerate(tfidf_louvain):
        name = "entropia_"+str(i)
        entropia_series = mat.apply(lambda x: nlp_utils.get_entropy(x), axis=1, raw=True).rename(name)
        entropias_louvain.append(entropia_series)
    
    entropias_infomap_df = pd.DataFrame(entropias_infomap).T
    entropias_louvain_df = pd.DataFrame(entropias_louvain).T

    results_infomap = pd.merge(infomap_clusters, entropias_infomap_df, left_on="comunidad",right_index=True).drop(columns=["comunidad"])
    results_louvain = pd.merge(louvain_clusters, entropias_louvain_df, left_on="comunidad",right_index=True).drop(columns=["comunidad"])
    
    return results_infomap, results_louvain

def random_entropy_iter(node_column,cluster_columns, vectorizers):
    corpus_infomap, corpus_louvain = build_random_corpus(node_column,cluster_columns)
    tfidf_infomap, tfidf_louvain = vectorize(corpus_infomap, corpus_louvain,vectorizers)
    results_infomap, results_louvain = cluster_entropy(tfidf_infomap,tfidf_louvain)
    results_infomap = results_infomap
    results_louvain = results_louvain
    return results_infomap, results_louvain

def random_cluster_entropy(iters,nodes,clusters,vectorizers):
    total_infomap = []
    total_louvain = []
    for i in range(iters):
        print(f"Iter {i}")
        infomap_results, louvain_results = random_entropy_iter(nodes,clusters,vectorizers)
        total_infomap.append(infomap_results)
        total_louvain.append(louvain_results)

    entropias_infomap = pd.concat(total_infomap)
    entropias_louvain = pd.concat(total_louvain)

    means_infomap = entropias_infomap.groupby("tamaño").mean().rename(columns={col:col+"_mean" for col in entropias_infomap.columns.values})
    std_infomap = entropias_infomap.groupby("tamaño").std().groupby("tamaño").mean().rename(columns={col:col+"_std" for col in entropias_infomap.columns.values})

    means_louvain = entropias_louvain.groupby("tamaño").mean().groupby("tamaño").mean().rename(columns={col:col+"_mean" for col in entropias_louvain.columns.values})
    std_louvain = entropias_louvain.groupby("tamaño").std().groupby("tamaño").mean().rename(columns={col:col+"_std" for col in entropias_louvain.columns.values})

    results_infomap = pd.concat([means_infomap,std_infomap],axis=1)
    results_louvain = pd.concat([means_louvain,std_louvain], axis=1)

    print("Done")

    return results_infomap,results_louvain

#%%
clusters = graph_node_data.loc[graph_node_data.degree_dd != 0,["comunidades_infomap","comunidades_louvain"]].reset_index(drop=True)
nodes = graph_node_data.loc[graph_node_data.degree_dd != 0, "node_index"].reset_index(drop=True)

args = nlp_args["TFIDF"]
stop_words = text.ENGLISH_STOP_WORDS.union(args["custom_stopwords"])
vectorizers = [TfidfVectorizer(tokenizer = None , stop_words=stop_words, ngram_range=ngram_range, min_df=args["min_df"],max_df=args["max_df"],max_features=args["max_features"])for ngram_range in [(1,1),(2,2),(3,3)]]

infomap_random_ids = infomap_clusters.sort_values(by="tamaño",ascending=False).drop_duplicates(subset="tamaño").comunidad.values
louvain_random_ids = louvain_clusters.sort_values(by="tamaño",ascending=False).drop_duplicates(subset="tamaño").comunidad.values

infomap_tamaños = infomap_clusters.tamaño.unique()
louvain_tamaños = louvain_clusters.tamaño.unique()

#%%

results_infomap, results_louvain = random_cluster_entropy(100,nodes,clusters,vectorizers)

results_infomap.to_csv(entropy_reports+"entropias_random_infomap.csv")
results_louvain.to_csv(entropy_reports+"entropias_random_louvain.csv")