# (1) Libraries ----
library(data.table)
library(glue)
library(tidyverse)
# (2) Load data ----
data_external = "../../data/external/"
data_interim = "../../data/interim/"
data_processed = "../../data/processed/"
kg_edges = fread(paste0(data_external,"primekg_edges.csv"), header=TRUE)
kg_d_features = fread(paste0(data_external, "primekg_disease_features.csv"), header=TRUE)
kg_nodes = fread(paste0(data_external,"primekg_nodes.csv"), header=TRUE)
mondo_cui_map = fread(paste0(data_interim,"mondo_cui_map.csv")) #mapeo que armé entre CUI y MONDO
# (3) def functions  ----
#Para armarme las tablas de enlaces y nodos de cada database
make_edge_dt = function(dt,relation,a_type,b_type,source){
  setnames(dt,colnames(dt),c("a_index","b_index","source_idx"))
  dt[,"relation"] = relation
  dt[,"a_type"] = a_type 
  dt[,"b_type"] = b_type
  dt[,"source"] = source
  return(dt)
}

make_node_dt = function(dt_src,dt_trg,src_type,trg_type,source){
  setnames(dt_src,old=colnames(dt_src),new=c("node_idx","node_name"))
  setnames(dt_trg,old=colnames(dt_trg),new=c("node_idx","node_name"))
  dt_src[,"node_type"] = src_type
  dt_trg[,"node_type"] = trg_type
  node_table = data.table(node_idx = integer(), node_type = character(), node_name = character())
  node_table = rbind(node_table,dt_src)
  node_table = rbind(node_table,dt_trg)
  node_table = unique(node_table, by="node_idx")
  node_table[,"node_source"] = source
  return (node_table)
}

# (5) Extraer redes ----
kg_edges <- tibble::rowid_to_column(kg_edges, "edge_index")
table(kg_edges$relation)
gda_edges = kg_edges[relation == "disease_protein",]
disease_edges = kg_edges[relation == "disease_disease",]

disease_edge_table = make_edge_dt(disease_edges[,c("x_index","y_index","edge_index")],"disease_disease","disease","disease","primekg")
gda_edge_table = make_edge_dt(gda_edges[,c("x_index","y_index","edge_index")],"gda","x","x","primekg")

###
mondo_cui_map$mondo = as.character(mondo_cui_map$mondo)
index_cui_mondo = merge(x=kg_nodes[node_source == "MONDO",c("node_index","node_id")],y=mondo_cui_map, by.x="node_id", by.y="mondo", all.x=TRUE)
setnames(index_cui_mondo,old=colnames(index_cui_mondo),new=c("mondo","kg_index","CUI"))
###
disease_node_table = kg_nodes[node_index %in% disease_edge_table$a_index | node_index %in% disease_edge_table$b_index,]
disease_node_table = merge(x=disease_node_table, y=index_cui_mondo, by.x="node_index", by.y="kg_index", all.x=TRUE)
###
gda_nodes = kg_nodes[node_index %in% gda_edge_table$a_index | node_index %in% gda_edge_table$b_index,]
gda_nodes = merge(x=gda_nodes, y=index_cui_mondo, by.x="node_index", by.y="kg_index", all.x=TRUE)
###
graph_edge_table = unique(rbind(gda_edge_table,disease_edge_table))
graph_node_table = unique(rbind(disease_node_table,gda_nodes,fill=TRUE))

write.csv(graph_node_table, file=paste0(data_interim,"primekg_exploring_node_table.csv"))
write.csv(graph_edge_table, file=paste0(data_interim,"primekg_exploring_graph_edge_table.csv"))
