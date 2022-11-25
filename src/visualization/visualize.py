#%%
from cluster_finder import find_cluster
from cluster_wordclouds import plot_wordcloud
from pathway_finder import find_pathways

#Examples:
find_cluster("infomap","nerve")
plot_wordcloud("infomap",443)
find_pathways("infomap",443)