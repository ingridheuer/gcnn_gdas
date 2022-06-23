#(1) Librerías ----
library(glue)
library(data.table)
#(2) Cargo datos ----
data_path = '/home/ingrid/Documents/tesis/datos_redes/'
system.time({primekg_edges = read.csv(paste0(data_path,"primekg/edges.csv"), header=TRUE)})
primekg_nodes = read.csv(paste0(data_path,"primekg/nodes.csv"), header=TRUE)
d_features = read.csv(paste0(data_path, "primekg/disease_features.csv"), header=TRUE)

library(data.table)
system.time({primekg_edges = fread(paste0(data_path,"primekg/edges.csv"), header=TRUE)})

#(3) ----
head(primekg_edges)
head(primekg_nodes)
unique(primekg_nodes$node_source)

#(4)Exploro Nodos ----
nodos_reactome = primekg_nodes[which(primekg_nodes$node_source == "REACTOME"),]
head(nodos_reactome)
unique(nodos$node_type)

unique(primekg_nodes$node_type)

genes_only = primekg_nodes[which(primekg_nodes$node_type == "gene/protein"),]
head(genes_only)
unique(genes_only$node_source) #todos los ids de genes/proteinas parecen ser de NCBI (entrez id)

disease_only = primekg_nodes[which(primekg_nodes$node_type == "disease"),]
head(disease_only)
disease_only[1,"node_id"] #chan

unique(disease_only$node_source)

aver = disease_only[which(disease_only$node_source == "MONDO"),] #los que tienen muchos ids raros es porque son los "grouped", los otros tienen un solo id de mondo
head(aver)
dim(aver)
dim(disease_only)

#(5) Exploro enlaces ----
#Veo que tipo de enlaces hay
unique(primekg_edges$relation)

#Solo los de enfermedad-enfermedad
disease_only = primekg_edges[which(primekg_edges$relation == "disease_disease"),]
head(disease_only)
dim(disease_only)
glue("Hay {dim(disease_only)[1]} enlaces enfermedad_enfermedad")
unique(disease_only$display_relation)

#Exploro features ----
head(d_features)
colnames(d_features)
dim(d_features)
con_complications = d_features[which(d_features$mayo_complications != ""),]
con_complications$mayo_complications[1]

#Exploro un ejemplo
ejemplo = d_features[10,"mondo_name"]
ejemplo
d_features[which(d_features$mondo_name == ejemplo), "umls_description"]
d_features[which(d_features$mondo_name == ejemplo), "group_id_bert"]
d_features[which(d_features$mondo_name == ejemplo), "mayo_symptoms"]
d_features[which(d_features$mondo_name == ejemplo), "orphanet_clinical_description"]

table(primekg_edges$display_relation)
primekg_edges[which(primekg_edges$display_relation == "side effect"),][1:10,]

table(A = signor$TYPEA, B = signor$TYPEB) #para ver dstintos tipos de relaciones que hay en signor 

# Probando datatables ----
primekg_edges = fread(paste0(data_path,"primekg/edges.csv"))
primekg_nodes = fread(paste0(data_path,"primekg/nodes.csv"))
table(primekg_nodes$node_type)
print("Hay 17080 nodos del tipo disease, yo obtuve solo mapeos CUI<->MONDO para 16876")

#Veo las sources
table(primekg_nodes[node_type == "disease", node_source])

# MAPEOS UMLS ----
mapping_table = fread(paste0(data_path, "mondo/mondo_cui_map.csv"))
head(mapping_table)
