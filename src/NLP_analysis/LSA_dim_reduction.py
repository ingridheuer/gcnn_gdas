#%%
import pandas as pd
import numpy as np
import random

from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

import config
import nlp_utils
#%%
seed = 16
random.seed(seed)
np.random.seed(seed)
#%%
data_processed = "../../data/processed/"
graph_data = data_processed + "graph_data_nohubs/"
lsa_data_path = graph_data+"LSA_data/"

args = config.nlp_args["LSA"]

#%%
def tfidf_to_lsa(sparse_dtm):
    # dense_dtm = sparse_dtm.sparse.to_dense()
    mat_index = sparse_dtm.index.astype(int).values

    print("Fitting LSA model ...")
    svd = TruncatedSVD(n_components=args["num_components"], random_state=seed)
    lsa_matrix = svd.fit_transform(sparse_dtm)

    vocab = sparse_dtm.columns.values
    component_data = {}

    print("Getting component vocab ...")
    for i, comp in enumerate(svd.components_):
        #Tuplas de cada término con su valor en esa componente
        vocab_comp = zip(vocab, comp)

        #Las ordeno según el valor de la componente, de mayor a menor, veo las primeras 10
        sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:10]
        wordlist = [pair[0] for pair in sorted_words]
        scorelist = [round(pair[1],3) for pair in sorted_words]
        component_data[i] = {"wordlist":wordlist,"scorelist":scorelist}

    component_vocab = pd.DataFrame.from_dict(component_data,orient="index")
    sparse_lsa = sparse.csr_matrix(lsa_matrix)

    print(f"Getting node similarity ...")
    similarity_matrix = np.triu(cosine_similarity(lsa_matrix,lsa_matrix),k=1)
    sparse_similarity_matrix = sparse.csr_matrix(similarity_matrix)

    return sparse_lsa,sparse_similarity_matrix, component_vocab, mat_index
#%%
print("Loading TFIDF matrices ...")
document_term_matrix = nlp_utils.load_node_matrices(graph_data+"tfidf_nodos/")
#%%
for i, dtm in enumerate(document_term_matrix):
    print(f"LSA {i} ...")
    lsa_matrix, similarity_matrix, component_vocab, mat_index = tfidf_to_lsa(dtm)

    print("Saving data ...")
    sparse.save_npz(f"{lsa_data_path}lsa_matrix_{i}.npz", lsa_matrix)
    sparse.save_npz(f"{lsa_data_path}similarity_matrix_{i}.npz", similarity_matrix)
    np.savetxt(f"{lsa_data_path}matrix_index_{i}.txt", mat_index)
    component_vocab.to_pickle(f"{lsa_data_path}component_vocab_{i}.pkl")


print(f"LSA done. Data saved to {lsa_data_path}")
#%%
