#%%
import nlp_utils
#%%
data_processed = "../../data/processed/"
graph_data = data_processed + "graph_data_nohubs/"
summary_path = "../../reports/summary/"

path_infomap = graph_data + "tfidf_infomap/"
path_louvain = graph_data + "tfidf_louvain/"
#%%
infomap_dtm, louvain_dtm = nlp_utils.load_cluster_matrices(path_infomap,path_louvain,1)

#%%
def get_truncated_matrices(full_matrix,N,name,path):
    top_N_terms = full_matrix.T.apply(lambda x: x.sort_values(ascending=False)[0:N].index).T
    top_N_terms.index = top_N_terms.index.astype(int)

    top_N_score = full_matrix.T.apply(lambda x: x.sort_values(ascending=False)[0:N].values).T
    top_N_score.index = top_N_score.index.astype(int)

    top_N_terms.to_csv(f"{path}top_{N}_terms_{name}.csv")
    top_N_score.to_csv(f"{path}top_{N}_score_{name}.csv")

#%%
get_truncated_matrices(infomap_dtm[0],100,"infomap",summary_path)
get_truncated_matrices(louvain_dtm[0],100,"louvain",summary_path)
