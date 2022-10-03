#(1) Librerías ----
library(graphite)
library(glue)
library(data.table)
#options(Ncpus = 2)
data_path = "/home/ingrid/Documents/tesis/datos_redes/"
disgenet = fread(paste0(data_path, "disgenet/curated_gene_disease_associations.tsv"))

#(2) Probando ---- 
humanReactome <- pathways("hsapiens", "reactome")
names(humanReactome)[1:10]  #me dice los pathways que tiene en reactome

#Instancio un pathway
p = humanReactome[["Apoptosis"]]

#Así pido la info del pathway
pathwayDatabase(p)
pathwayTimestamp(p)

#Pido los nodos
nodes(p)
head(edges(p))
databases = pathwayDatabases()
#pathways de hsapiens:
databases[which(databases$species == "hsapiens"),2]
print("Son 7 bases de datos, voy a ir armando tablas y después las junto todas")

# (3) Reactome ----
glue("Reactome almacena {length(names(humanReactome))} pathways")
names(humanReactome)

#Pruebo mapear los genes de un pathway a los de disgenet
p1 = humanReactome[["Sodium/Calcium exchangers"]]
p1_entrez = convertIdentifiers(p1, "ENTREZID")
glue("El pathway \"{pathwayTitle(p1)}\" tiene {length(nodes(p1))} nodos y {dim(edges(p1))[1]} enlaces \n Los enlaces son de tipo {unique(edges(p1)[,6])}")
glue("Al convertirlo a entrez ID tiene {length(nodes(p1_entrez))} nodos y {dim(edges(p1_entrez))[1]} enlaces")
#por qué me aumenta la cantidad de nodos/enlaces?

p1_df = edges(p1_entrez)
p1_uniprot_df = edges(p1)
p1_uniprot_df
p1_df

# (4) KEGG ----
kegg = pathways("hsapiens", "kegg")
names(kegg)
p2 = kegg[["Retinol metabolism"]]
p3 = kegg[[5]]
edges(p3)
print("KEGG ya está mapeado a entrez id")

# (5) PANTHER ----
panther = pathways("hsapiens", "panther")
panther
names(panther)
p4 = panther[["Ras Pathway"]]
edges(p4)
aver = convertIdentifiers(p4, "ENTREZID")
edges(aver)
p4

#(6) Tablas - reactome (probando) ----
reactome = pathways("hsapiens","reactome")
disgenet_genes = unique(disgenet[,"geneId"])

p1 = reactome[[1]]
p1_entrez = convertIdentifiers(p1, "ENTREZID")
geneids = lapply(nodes(p1_entrez), function(x){gsub("ENTREZID:","",x)})
geneids = as.integer(geneids)

#Es una tabla de #genes * #pathways, solo con reactome eso da 9703*2439 = 23665617 (no es un monton?)
disgenet_genes[,pathwayId(p1)] = disgenet_genes[,as.integer(geneId %in% geneids)] #funciono!!

#Ahora tengo que repetir esto para todos los pathways basicamente
#Armo una tabla con solo los ids de genes como columna por ahora
gene_pathways = unique(disgenet[,"geneId"])
table(disgenet_genes$`R-HSA-1059683`)

#----
#Itero sobre todos los pathways de reactome
#esto funcionó pero nosé que pasa con los mapeos "1:many" y los NA que dice que tuvo que forzar (no los encuentro en el datatable) -> cuando no tiene mapeo deja el identificador y esto me daba error
for (pathway in as.list(reactome)){
    pathway = convertIdentifiers(pathway,"ENTREZID")
    nodes = nodes(pathway)[grepl("ENTREZID:", nodes(pathway))] #me quedo solo con los que tienen entrez id, sino después quedan strings y no puedo usar as integer
    geneids = lapply(nodes, function(x){gsub("ENTREZID:","",x)}) #saco el "ENTREZID:"
    geneids = as.integer(geneids)
    gene_pathways[,pathwayId(pathway)] = gene_pathways[,as.integer(geneId %in% geneids)]
}

#Intento esto con un try para que no me corte el loop por el error
create_table = function(pathway){
  pathway = convertIdentifiers(pathway,"ENTREZID",)
  nodes = nodes(pathway)[grepl("ENTREZID:", nodes(pathway))] #me quedo solo con los que tienen entrez id, sino después quedan strings y no puedo usar as integer
  geneids = lapply(nodes, function(x){gsub("ENTREZID:","",x)}) #saco el "ENTREZID:"
  geneids = as.integer(geneids)
  gene_pathways[,pathwayId(pathway)] = gene_pathways[,as.integer(geneId %in% geneids)]
}

for (pathway in as.list(reactome)){
  try(create_table(pathway))
}

#veo que pasa con los que salen "1:many mapping"
p2 = reactome[[2]]
p2_entrez = convertIdentifiers(p2, "ENTREZID")
length(nodes(p2))
length(nodes(p2_entrez))
#conclusión: no se

#Veos si puedo usar la nueva tabla para obtener los pathways que pertenece un gen
#Lista de pathways a los que pertenece el gen con id 1:
which(gene_pathways == 0 & gene_pathways$geneId == 1, arr.ind=TRUE)
table(gene_pathways[geneId == 1,])
dim(gene_pathways)
nodes = nodes(p1_entrez)[grepl("ENTREZID:",nodes(p1_entrez))]
nodes

title = "Uncoating of the HIV Virion"
reactome[[title]]
aver = convertIdentifiers(reactome[[title]], "ENTREZID")
