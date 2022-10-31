#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import plotly.io as pio
#%%
cmap = plt.get_cmap("tab10")
pio.templates.default = "seaborn"

sns.set_style("darkgrid", rc={'xtick.bottom': True})
sns.set_context("paper")
#%%
data_processed = "../../data/processed/"
data_interim = "../../data/interim/"
data_external = "../../data/external/"
tfidf_reports = "../../reports/tfidf/"

path_infomap = tfidf_reports+"entropia_random_infomap_mono.txt"
path_louvain = tfidf_reports+"entropia_random_louvain_mono.txt"
#%%
entropia_random_infomap = 1 - np.loadtxt(path_infomap)
entropia_random_louvain = 1 - np.loadtxt(path_louvain)

means_random_infomap = [np.mean(arr) for arr in np.array(entropia_random_infomap).T]
stds_random_infomap = [np.std(arr) for arr in np.array(entropia_random_infomap).T]

means_random_louvain = [np.mean(arr) for arr in np.array(entropia_random_louvain).T]
stds_random_louvain = [np.std(arr) for arr in np.array(entropia_random_louvain).T]
#%%
infomap_clusters = pd.read_pickle(tfidf_reports+"infomap_analysis_checkpoint.pkl")
louvain_clusters = pd.read_pickle(tfidf_reports+"louvain_analysis_checkpoint.pkl")
#%%
ids_random_infomap = infomap_clusters.drop_duplicates(subset="tamaño").comunidad.values
tamaños_random_infomap = infomap_clusters.set_index("comunidad").loc[ids_random_infomap,"tamaño"].values

ids_random_louvain = louvain_clusters.drop_duplicates(subset="tamaño").comunidad.values
tamaños_random_louvain = louvain_clusters.set_index("comunidad").loc[ids_random_louvain,"tamaño"].values
#%%
infomap_clusters_temp = infomap_clusters.copy()
infomap_clusters_temp["spec"] = 1 - infomap_clusters_temp.entropia_1
infomap_clusters_mean_s = infomap_clusters_temp[["tamaño","spec"]].groupby(["tamaño"]).mean().fillna(0)
infomap_clusters_std_s = infomap_clusters_temp[["tamaño","spec"]].groupby(["tamaño"]).std().fillna(0)

louvain_clusters_temp = louvain_clusters.copy()
louvain_clusters_temp["spec"] = 1 - louvain_clusters_temp.entropia_1
louvain_clusters_mean_s = louvain_clusters_temp[["tamaño","spec"]].groupby(["tamaño"]).mean().fillna(0)
louvain_clusters_std_s = louvain_clusters_temp[["tamaño","spec"]].groupby(["tamaño"]).std().fillna(0)
#%%
x1, y1, std1 = infomap_clusters_mean_s.index.values, infomap_clusters_mean_s.spec.values, infomap_clusters_std_s.spec.values
x1_random, y1_random, std1_random = tamaños_random_infomap, np.array(means_random_infomap),  np.array(stds_random_infomap)

fig, ax = plt.subplots(1,2,figsize=(9,4), sharex=True, sharey=True)
# fig.suptitle("Semantic specificity of disease clusters vs cluster size\n",fontsize=18)
ax[0].scatter(x1 , y1, linewidths=0.3, c="b",linestyle='None', label="Mean specificity, Infomap clusters")
ax[0].errorbar(x1, y1,std1, color='b',linestyle='None', alpha=0.2)

ax[0].plot(x1_random , y1_random, "r--", label="Mean specificity, control sample")
ax[0].fill_between(x1_random, y1_random - std1_random, y1_random + std1_random, color='r', alpha=0.2)

ax[0].set_title("Infomap")
ax[0].grid(True)
ax[0].set_xlabel("Number of nodes")
ax[0].set_ylabel("Specificity")
ax[0].set_xscale("log")
ax[0].legend()


x2, y2, std2 = louvain_clusters_mean_s.index.values, louvain_clusters_mean_s.spec.values, louvain_clusters_std_s.spec.values
x2_random, y2_random, std2_random = tamaños_random_louvain, np.array(means_random_louvain), np.array(stds_random_louvain)
ax[1].scatter(x2 , y2, linewidths=0.3, c="b",linestyle='None', label="Mean specificity, Louvain clusters")
ax[1].errorbar(x2, y2, std2, color='b',linestyle='None', alpha=0.2)

ax[1].plot(x2_random , y2_random, "r--", label="Mean specificity, control sample")
ax[1].fill_between(x2_random, y2_random - std2_random, y2_random + std2_random, color='r', alpha=0.2)

ax[1].set_title("Louvain")
ax[1].grid(True)
ax[1].set_xlabel("Number of nodes")
ax[1].set_ylabel("Specificity")
ax[1].set_xscale("log")
ax[1].legend()
fig.tight_layout()
#%%
fig = px.scatter(louvain_clusters_temp, x="spec", y="mean_similarity_mono_triu", size="tamaño",width=800, height=600,marginal_x="histogram",marginal_y="histogram")
fig.update_layout(xaxis_title="Specificity", yaxis_title="Mean TF-IDF similarity")

fig.show()
#%%

g = sns.jointplot(data=louvain_clusters_temp, x="spec", y="mean_similarity_mono_triu",height=7)
g.ax_joint.cla()
sns.scatterplot(data=louvain_clusters_temp, x="spec", y="mean_similarity_mono_triu", size="tamaño",ax=g.ax_joint,sizes=(1, 200),legend=False,alpha=0.6)
g.set_axis_labels('Specificity', 'Mean TF-IDF similarity', fontsize=12)

#%%
fig, ax = plt.subplots()
sns.histplot(data=louvain_clusters.tamaño, log_scale=True,color=sns.color_palette()[1],bins=25)
ax.set_xlabel("Cluster size", fontsize=12)
#%%