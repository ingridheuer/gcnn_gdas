#%%
import pandas as pd
import numpy as np
import random
import regex as re
import pickle

from sklearn.feature_extraction import text
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

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

graph_node_data = pd.read_csv(graph_data+"nohub_graph_node_data.csv")
disease_attributes = pd.read_csv(graph_data+"nohub_disease_attributes.csv")

#agrego los nodos bert al df de disease attributes (aunque no tienen atributos)
nodos_bert = graph_node_data.loc[graph_node_data.node_type == "bert_group",["node_index","node_id","node_name","node_source"]].copy()
disease_attributes = pd.concat([disease_attributes,nodos_bert])

enfermedades_en_dd = graph_node_data.loc[graph_node_data.degree_dd != 0, "node_index"].values

#me quedo solo con las enfermedades que están en DD
disease_attributes = disease_attributes.set_index("node_index").loc[enfermedades_en_dd].reset_index()
disease_attributes = pd.merge(graph_node_data[["node_index","comunidades_infomap","comunidades_louvain"]],disease_attributes,left_on="node_index",right_on="node_index",how="right")

custom_stopwords = config.nlp_args["TFIDF"]["custom_stopwords"]
#%%
stop_words = text.ENGLISH_STOP_WORDS.union(custom_stopwords)

def remove_symbols(data):
    symbols = "!-.\"\'#$%&()*+/:;<=>?@[\]^_`{|}°,~\n"
    for i in symbols:
        data = np.char.replace(data, i, ' ')
    return data

def remove_numbers(text,any_number,whitespace):
    if any_number:
        text = re.sub(r'\d+', '', text)
    if whitespace:
        text = re.sub(r'\b(\d+)\b', '', text)
    return text

def more_regex_cleaning(text):
    patterns = [r'(\d+)hr',r'(\d+)th',r'(\d+)cm',r'(\d+)mm',r'(\d+)mmhg',r'\b[1-9][a-zA-Z]\b',r'\b[a-zA-Z][1-9]\b',r'(\d+)a',r'(\d+)b',r'(\d+)c']
    for pattern in patterns:
        text = re.sub(pattern, "", text)
    return text

lemmatizer = WordNetLemmatizer()
def getWordNetPOS(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tagDict = {"J": wordnet.ADJ,
    "N": wordnet.NOUN,
    "V": wordnet.VERB,
    "R": wordnet.ADV}
    return tagDict.get(tag, wordnet.NOUN)

def lemmatize_text(lemmatizer,text):
    words = text.split()
    new_text = ""
    tags = [getWordNetPOS(word) for word in words]
    for word,tag in zip(words,tags):
        lem = lemmatizer.lemmatize(word,pos=tag)
        new_text += " " + lem 
    return new_text

def filter_stopwords(text):
    new_text = ""
    words = text.split()
    for word in words:
        if word not in stop_words:
            new_text = new_text + " " + word
    return new_text

def preprocess(corpus,lemmatizer,lemma_filter,remove_stopwords=True,sub_numbers=False,regex_clean=True):
    corpus = np.char.lower(corpus)
    corpus = remove_symbols(corpus)
    if remove_stopwords:
        corpus = np.array([filter_stopwords(text) for text in corpus]).astype(str)
    if lemma_filter:
        corpus = np.array([lemmatize_text(lemmatizer,text) for text in corpus]).astype(str)
    if sub_numbers:
        corpus = np.array([remove_numbers(text,any_number=False,whitespace=True) for text in corpus]).astype(str)
    if regex_clean:
        corpus = np.array([more_regex_cleaning(text) for text in corpus]).astype(str)
    if remove_stopwords:
        corpus = np.array([filter_stopwords(text) for text in corpus]).astype(str)

    return corpus


def node_as_document(node_index,df,lemmatizer,lemma_filter=True,remove_stopwords=True,sub_numbers=True,join_titles=True):
    data = df.loc[df["node_index"] == node_index, ["node_name","mondo_definition","umls_description","orphanet_definition"]].values.astype(str)
    data = np.delete(data,np.where(data[0] == "nan"))
    data = preprocess(data,lemmatizer,lemma_filter,remove_stopwords,sub_numbers)
    if join_titles:
        document = " ".join(data)
        document = document + " "
        document = document.replace(". ", " ")
    else:
        document = data
    return document


def node_name_as_document(node_index,df,lemmatizer,lemma_filter=True,remove_stopwords=True,sub_numbers=True,join_titles=True):
    data = df.loc[df["node_index"] == node_index, "node_name"].values.astype(str)
    data = preprocess(data,lemmatizer,lemma_filter,remove_stopwords,sub_numbers)
    if join_titles:
        document = " ".join(data)
        document = document + " "
        document = document.replace(". ", " ")
    else:
        document = data
    return document
#%%
print("Processing node documents ...")
processed_node_documents = {node_index:node_as_document(node_index,disease_attributes,lemmatizer) for node_index in disease_attributes.node_index.values}

#%%
with open(graph_data+"processed_node_documents.pickle", "wb") as handle:
    pickle.dump(processed_node_documents, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Processing done. Proccesed data saved at {graph_data}+processed_node_documents.pickle")
# %% 
