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
hippie = fread(paste0(data_external,"hippie_current.txt"))
signor = fread(paste0(data_interim,"signor_mapped.csv")) #signor con mapeo que hice a entrez ID
disgenet_gdas = fread(paste0(data_external,"curated_gene_disease_associations.tsv"), header = TRUE)
disgenet_attr = fread(paste0(data_external,"disease_mappings_to_attributes.tsv"), header= TRUE)

setnames(
  hippie,
  old = colnames(hippie),
  new = c(
    "P1_NAME",
    "P1_ENTREZ_ID",
    "P2_NAME",
    "P2_ENTREZ_ID",
    "SCORE",
    "EVIDENCE"
  )
)
# (3) def functions  ----
#Para armarme las tablas de enlaces y nodos de cada database
make_edge_dt = function(dt,relation,a_type,b_type,source){
  setnames(dt,colnames(dt),c("a_id","b_id","source_idx"))
  dt[,"relation"] = relation
  dt[,"a_type"] = a_type
  dt[,"b_type"] = b_type
  dt[,"source"] = source
  return(dt)
}

make_node_dt = function(dt_src,dt_trg,src_type,trg_type,source){
  setnames(dt_src,old=colnames(dt_src),new=c("node_id","node_name"))
  setnames(dt_trg,old=colnames(dt_trg),new=c("node_id","node_name"))
  dt_src[,"node_type"] = src_type
  dt_trg[,"node_type"] = trg_type
  node_table = data.table(node_id = character(), node_type = character(), node_name = character())
  node_table = rbind(node_table,dt_src)
  node_table = rbind(node_table,dt_trg)
  node_table = unique(node_table, by="node_id")
  node_table[,"node_source"] = source
  return (node_table)
}

#Para ver si quedaron NaNs y duplicados en las tablas
tests = function(node_dt,edge_dt){
  print("NAs in node table")
  print(table(is.na(node_dt[,"node_id"])))
  print("Duplicated in node table")
  print(table(duplicated(node_dt,by="node_id")))
  print("NAs in edge table ids")
  print(table(is.na(edge_dt[,c("a_id","b_id")])))
  print("NAs in edge table all")
  print(table(is.na(edge_dt)))
  print("Duplicated in edge table")
  print(table(duplicated(edge_dt,by=c("a_id","b_id"))))
}

#para obtener los nodos y enlaces duplicados
get_edge_dups = function(edge_dt){
  return (edge_dt[duplicated(edge_dt,by=c("a_id","b_id")) | duplicated(edge_dt, by=c("a_id","b_id"), fromLast=TRUE),])
}

get_node_dups = function(node_dt){
  return (node_dt[duplicated(node_dt,by="node_id") | duplicated(node_dt, by="node_id", fromLast=TRUE),])
}

get_edge_dups_2 = function(edge_dt,a_col,b_col){
  return (edge_dt[duplicated(edge_dt,by=c(a_col,b_col)) | duplicated(edge_dt, by=c(a_col,b_col), fromLast=TRUE),])
}

# (4) ----
graph_edge_table = data.table(a_id = character(), b_id = character(), relation = character(), source = character(), source_idx = integer(),a_type = character(), b_type = character())
graph_node_table = data.table(node_id = character(), node_type = character(), node_name = character(), node_source = character()) 
# (5) edges de disgenet  ----

#agrego una columna con un indice para cada enlace
disgenet_edges <- tibble::rowid_to_column(disgenet_gdas, "edge_index")
disgenet_edge_table = make_edge_dt(disgenet_edges[,c("geneId","diseaseId","edge_index")],"GDA","gene/protein","disease","disgenet")
disgenet_edge_table = merge(disgenet_edge_table, disgenet_edges[,c("edge_index","YearInitial","YearFinal","score")], by.x="source_idx",by.y="edge_index",all.y=FALSE,all.x=TRUE)

#para armar la tabla de nodos agrego la columna de source y la de target, después saco los duplicados
disgenet_src_nodes = disgenet_gdas[,c("geneId","geneSymbol")]
disgenet_trg_nodes = disgenet_gdas[,c("diseaseId","diseaseName")]
disgenet_node_table = make_node_dt(disgenet_src_nodes,disgenet_trg_nodes,"gene/protein","disease","disgenet")
disgenet_node_table = merge(disgenet_node_table, disgenet_attr[,c("diseaseId","type","diseaseClassMSH","diseaseClassNameMSH")], by.x="node_id", by.y="diseaseId", all.y=FALSE,all.x=TRUE)
setnames(disgenet_node_table,"type","disgenet_type")

tests(disgenet_node_table,disgenet_edge_table)
#van a quedar NAs en edge table porque algunos datos como yearinitial, yearfinal y attrs de enfermedades nestán vacíos

# (6) edges de hippie ----
hippie_edges <- tibble::rowid_to_column(hippie, "edge_index")
hippie_edge_table = make_edge_dt(hippie_edges[SCORE >= 0.73 ,c("P1_ENTREZ_ID","P2_ENTREZ_ID","edge_index")],"PPI","gene/protein","gene/protein","hippie")
hippie_edge_table = merge(hippie_edge_table, hippie_edges[,c("edge_index","SCORE")], by.x="source_idx",by.y="edge_index", all.y=FALSE, all.x=TRUE)
setnames(hippie_edge_table, "SCORE","score")

hippie_src_nodes = hippie[,c("P1_ENTREZ_ID","P1_NAME")]
hippie_trg_nodes = hippie[,c("P2_ENTREZ_ID","P2_NAME")]
hippie_node_table = make_node_dt(hippie_src_nodes,hippie_trg_nodes,"gene/protein","gene/protein","hippie")

tests(hippie_node_table, hippie_edge_table)

#veo todos los duplicados (uso este argumento para que me de todos los casos y no solo los que son "duplicado de")
hippie_dups = hippie_edge_table[duplicated(hippie_edge_table,by=c("a_id","b_id")) | duplicated(hippie_edge_table, by=c("a_id","b_id"), fromLast=TRUE),]

#me fijo si source_idx está duplicado, si no está duplicado es porque ya estaba repetido en el dataset original
table(duplicated(hippie_dups[,source_idx]))

hippie[hippie_dups[,source_idx],]

#Veo un ejemplo
hippie_dups_ex = hippie_dups[a_id == 3123 & b_id == 3122, source_idx]
hippie[hippie_dups_ex,]
#son genes con mismo ID pero distinto nombre! Me quedo con un único caso

hippie_edge_table = unique(hippie_edge_table, by=c("a_id","b_id"))
tests(hippie_node_table,hippie_edge_table)

# (7) edges de signor - PPI ----
#no hago drop na primero porque pierdo los que no están mapeados (y no estan mapeados porque son complejos)
signor_edges <- tibble::rowid_to_column(signor, "edge_index")
signor_ppi_edge_table = signor_edges[TYPEA == "protein" & TYPEB == "protein",c("ENTREZ_ID_A","ENTREZ_ID_B","edge_index")]
signor_ppi_edge_table = signor_ppi_edge_table[complete.cases(signor_ppi_edge_table)]
signor_ppi_edge_table = make_edge_dt(signor_ppi_edge_table[,c("ENTREZ_ID_A","ENTREZ_ID_B","edge_index")],"PPI","gene/protein","gene/protein","signor")
signor_ppi_edge_table = merge(signor_ppi_edge_table, signor_edges[,c("edge_index","SCORE")], by.x="source_idx",by.y="edge_index", all.y=FALSE,all.x=TRUE)
setnames(signor_ppi_edge_table, "SCORE","score")

signor_src_nodes_ppi = signor[TYPEA == "protein" & TYPEB == "protein",c("ENTREZ_ID_A","ENTITYA")]
signor_trg_nodes_ppi = signor[TYPEA == "protein" & TYPEB == "protein",c("ENTREZ_ID_B","ENTITYB")]

signor_ppi_node_table = make_node_dt(signor_src_nodes_ppi,signor_trg_nodes_ppi,"gene/protein","gene/protein","signor")

tests(signor_ppi_node_table, signor_ppi_edge_table)
signor_ppi_node_table = signor_ppi_node_table[complete.cases(signor_ppi_node_table),]

signor_dups = get_edge_dups(signor_ppi_edge_table)
table(duplicated(signor_dups$source_idx)) #estaban duplicados en el original, por ahora los saco
signor_ppi_edge_table = unique(signor_ppi_edge_table, by=c("a_id","b_id"))

tests(signor_ppi_node_table,signor_ppi_edge_table)
#(8) edges de signor- complex-protein ----
signor_cp_edges = signor_edges[TYPEA == "protein" & TYPEB == "complex" & EFFECT == "form complex", c("ENTREZ_ID_A","IDB","edge_index")]
signor_cp_edges = signor_cp_edges[complete.cases(signor_cp_edges),]
signor_cp_edge_table = make_edge_dt(signor_cp_edges,"forms_complex","gene/protein","protein_complex","signor")

signor_src_nodes_cp = signor[TYPEA == "protein" & TYPEB == "complex" & EFFECT == "form complex", c("ENTREZ_ID_A","ENTITYA")]
signor_trg_nodes_cp = signor[TYPEA == "protein" & TYPEB == "complex" & EFFECT == "form complex", c("IDB","ENTITYB")]
signor_src_nodes_cp = signor_src_nodes_cp[complete.cases(signor_src_nodes_cp),]

signor_cp_node_table = make_node_dt(signor_src_nodes_cp, signor_trg_nodes_cp, "gene/protein", "protein_complex", "signor")

tests(signor_cp_node_table,signor_cp_edge_table)
get_edge_dups(signor_cp_edge_table)
table(duplicated(get_edge_dups(signor_cp_edge_table))) #como antes, estaban duplicados en el dataset

signor_cp_edge_table = unique(signor_cp_edge_table,by=c("a_id","b_id"))

tests(signor_cp_node_table,signor_cp_edge_table)

#(9) edges de signor complex-protein -> estos no los poniamos porque es otro tipo de enlace ----
signor_pc_edges = signor[TYPEA == "complex" & TYPEB == "protein"]
table(signor_pc_edges$MECHANISM)

# (10) edges disease-disease de prime kg ----
#Primero tengo que mapear los ids de mondo a los de cui con el mapping table que me había armado
#Me voy a quedar solo con las enfermedades que pude mapear
kg_edges <- tibble::rowid_to_column(kg_edges, "edge_index")
kg_single_disease_nodes = kg_nodes[node_type == "disease" & node_source == "MONDO"]
kg_grouped_disease_nodes = kg_nodes[node_type == "disease" & node_source == "MONDO_grouped"] #acá son grupos de ids separados por "_"
kg_single_disease_nodes = kg_single_disease_nodes[, node_id:=as.integer(node_id)] #ahora que solo tengo ids mondo que son números puedo hacer esto

kg_single_disease_nodes_mapped =
  merge(x = kg_single_disease_nodes,
        y = mondo_cui_map,
        by.x = "node_id",
        by.y = "mondo",
        all.x = FALSE)[]
kg_single_disease_nodes_mapped
glue("En el mapeo pierdo {dim(kg_single_disease_nodes)[1] - dim(kg_single_disease_nodes_mapped)[1]} nodos porque no está el mapeo CUI-mondo")

CUI_kg_id_map = kg_single_disease_nodes_mapped[,c("node_index","CUI")]

kg_dd_edges = kg_edges[relation == "disease_disease",c("x_index","y_index","edge_index")] #son todos display_relation = parent-child
kg_dd_edges = merge(kg_dd_edges, CUI_kg_id_map, by.x = "x_index", by.y = "node_index", all.x = FALSE) #pierdo los que no tienen mapeo CUI
setnames(kg_dd_edges, old = "CUI", new = "x_CUI")

kg_dd_edges = merge(kg_dd_edges, CUI_kg_id_map, by.x = "y_index", by.y = "node_index", all.x = FALSE) #pierdo los que no tienen mapeo CUI
setnames(kg_dd_edges, old = "CUI", new = "y_CUI")

#Ahora si agrego los edges a mi dataset
kg_dd_edge_table = make_edge_dt(kg_dd_edges[,c("x_CUI","y_CUI","edge_index")],"parent_child_mondo","disease","disease","primekg")

kg_node_table = kg_single_disease_nodes_mapped[,c("CUI","node_name")]
setnames(kg_node_table, old=colnames(kg_node_table),new=c("node_id","node_name"))
kg_node_table[,"node_source"] = "primekg"
kg_node_table[,"node_type"] = "disease"

tests(kg_node_table,kg_dd_edge_table)

dup_ids = get_node_dups(kg_node_table)[,node_id]
dup_maps = mondo_cui_map[CUI %in% dup_ids,]
#Veo que el mismo CUI mapea a varios mondo, por eso hay dups, por ahora me quedo con los unicos
dup_maps[order(dup_maps$CUI),]

kg_node_table = unique(kg_node_table,by="node_id")
kg_edge_table = unique(kg_dd_edge_table,by=c("a_id","b_id"))

tests(kg_node_table,kg_edge_table)

# (11) Junto todos los dataframes  ----
graph_node_table = do.call("rbind", list(disgenet_node_table,hippie_node_table,signor_ppi_node_table,signor_cp_node_table,kg_node_table, fill=TRUE))
graph_edge_table = do.call("rbind", list(disgenet_edge_table,hippie_edge_table,signor_ppi_edge_table,signor_cp_edge_table,kg_edge_table, fill=TRUE))

tests(graph_node_table,graph_edge_table)
#probablemente porque el mismo nodo aparece en varios datasets, me puedo quedar con un solo caso para los nodos

graph_edge_dups = get_edge_dups(graph_edge_table)
graph_node_dups = get_node_dups(graph_node_table)

graph_node_dups[order(node_id)] #efectivamente, el mismo nodo aparece en distintos datasets -> para los repetidos disgenet, conservo los disgenet
graph_edge_dups[order(a_id)] #idem, el mismo enlace aparece en distintos datasets, si es el mismo tipo de relacion me puedo quedar con uno solo

#NODOS
#como es el primero que agregué, siempre debería aparecer primero disgenet, así que los duplicates....
table(graph_node_table[duplicated(graph_node_table,by="node_id"),node_source]) 

#son todos de otros datasets, probablemente aparecen primero en disgenet, me quedo con los primeros
graph_node_table = unique(graph_node_table,by="node_id")

#ENLACES
table(graph_edge_dups[,source]) #obvio que los repetidos son hippie o signor porque son los unicos que comparten un tipo de enlace (PPI)
table(graph_edge_table[duplicated(graph_edge_table,by=c("a_id","b_id")),source]) #como agregué primero hippie, si uso unique me quedan los de hippie

graph_edge_table = unique(graph_edge_table, by=c("a_id","b_id"))

tests(graph_node_table,graph_edge_table)

# (12) Por último genero indice interno para los edges y nodos en mi grafo ----
graph_node_table = tibble::rowid_to_column(graph_node_table, "node_idx")

#Agrego los indices internos a la tabla de enlaces
graph_edge_table = merge(graph_edge_table,graph_node_table[,c("node_id","node_idx")],by.x="a_id",by.y="node_id")
setnames(graph_edge_table,old="node_idx",new="a_idx")
graph_edge_table = merge(graph_edge_table,graph_node_table[,c("node_id","node_idx")],by.x="b_id",by.y="node_id")
setnames(graph_edge_table,old="node_idx",new="b_idx")

tests(graph_node_table,graph_edge_table)
graph_edge_table = graph_edge_table[,c("a_idx","b_idx","a_id","b_id","relation","a_type","b_type","source","source_idx","score","YearInitial","YearFinal")][order(source)]
#graph_edge_table = graph_edge_table[,c("a_idx","b_idx","a_id","b_id","relation","a_type","b_type","source","source_idx")][order(source)]
graph_edge_table = tibble::rowid_to_column(graph_edge_table, "edge_idx")

graph_node_table
graph_edge_table

#Guardo los dataframes
write.csv(graph_node_table, file=paste0(data_processed,"graph_node_table.csv"))
write.csv(graph_edge_table, file=paste0(data_processed,"graph_edge_table.csv"))

#Armo tabla con tipos de enlace y directed=0 si es no dirigido, directed=1 si es dirigido
edge_types = unique(graph_edge_table$relation)
directed = c(0,0,0,0)
df = data.frame(edge_types,directed)
write.csv(df, file=paste0(data_processed,"edgetype_directed.csv"))

# (13) Exploro como quedó el dataset *pasar esto a un notebook o un script de la carpeta exploration ----
# glue("El dataset tiene {dim(graph_node_table)[1]} nodos y {dim(graph_edge_table)[1]} enlaces")
# 
# print("Nodos agrupados por tipo:")
#table(graph_node_table$node_type)
# 
# print("Nodos agrupados por fuente: (conservé todos los de disgenet)")
# table(graph_node_table$node_source)
# dim(disgenet_node_table)
# 
# print("Enlaces agrupados por tipo de enlace:")
# table(graph_edge_table$relation)
# 
# print("Enlaces agrupados por tipos de nodo")
# table(graph_edge_table[,c("a_type","b_type")])
# 
# print("Enlaces agrupados por fuente: prioridad PPIs HIPPIE, es decir, saqué ppis de signor que ya estaban en hippie")
# table(graph_edge_table$source)
# dim(disgenet_gdas)
# dim(hippie)
# dim(signor)
# 
# dim(kg_edges[relation == "disease_disease",])
# dim(kg_grouped_disease_nodes)
# dim(kg_single_disease_nodes)
# dim(kg_nodes[node_type == "disease"])
# print("PrimeKG tiene nodos enfermedad que son grupos obtenidos con BERT, esos no los agregué")
# glue("Además en el mapeo perdí {dim(kg_single_disease_nodes)[1] - dim(kg_single_disease_nodes_mapped)[1]} nodos porque no están en el mapeo CUI-mondo")
# 
# print("Enlaces de primekg que involucraban a nodos *grupo* de enfermedades")
# idx_grouped = kg_grouped_disease_nodes[,node_index]
# dim(kg_edges[relation == "disease_disease" & (x_index %in% idx_grouped | y_index %in% idx_grouped) ,])
# 
# print("Self loops")
# dim(graph_edge_table[a_idx == b_idx])
# 
# print("Los self loops son de:")
# table(graph_edge_table[a_idx == b_idx, source])
# graph_edge_table[a_idx == b_idx & source == "primekg",source_idx]
# 
# print("Nodos que me quedaron sin enlaces")
# nodos_desconectados = graph_node_table[!(node_idx %in% graph_edge_table[,a_idx]) & !(node_idx %in% graph_edge_table[,b_idx]),]
# nodos_desconectados
# print("Son todos nodos de primekg y signor, revisar esto")
# table(nodos_desconectados[,node_source])
# 
# #veo un ejemplo de los que quedaron afuera
# table(kg_edges[(x_index == 40792 | y_index == 40792), relation])
