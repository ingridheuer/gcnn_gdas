# (1) Librerías ----
library(glue)
library(data.table)
# (2) Datos ----
data_path = "/home/ingrid/Documents/tesis/datos_redes/"
disgenet = fread(paste0(data_path, "disgenet/curated_gene_disease_associations.tsv"))
hippie = fread(paste0(data_path,"hippie_current_v2.3.txt"))
#(3) HIPPIE ----
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

#Saco los "_HUMAN" de los nombres, comparo lo que tarda dataframe vs data table
system.time({hippie$P1_NAME = lapply(hippie$P1_NAME, function(x) {gsub("_HUMAN", "", x)})})
#system.time({hippie[, P1_NAME := lapply(P1_NAME, function(x) {gsub("_HUMAN","",x)}), P1_NAME]})
system.time({hippie[, P2_NAME := lapply(P2_NAME, function(x) {gsub("_HUMAN","",x)}), P2_NAME]})


#(4) Exploro HIPPIE ----
ids_hippie = unique(c(hippie$P1_ENTREZ_ID, hippie$P2_ENTREZ_ID))
ids_disgenet = unique(disgenet$geneId)
glue("Hay {length(ids_hippie)} proteínas distintas en HIPPIE y {length(ids_disgenet)} genes distintos en DisGeNET (curado)")
glue("Hay {dim(hippie)[1]} enlaces en HIPPIE y {dim(disgenet)[1]} enlaces en DisGeNET (curado)")
print("Proteínas de hippie que están y no están en disgenet:")
table(ids_hippie %in% ids_disgenet)

print("Interacciones de HIPPIE donde ambas proteínas están en disgenet")
table(hippie[,P1_ENTREZ_ID] %in% ids_disgenet & hippie[,P2_ENTREZ_ID] %in% ids_disgenet)

#Si me quiero quedar solo con los enlaces entre proteinas que están en disgenet
hippie_in_disg = hippie[P1_ENTREZ_ID %in% ids_disgenet & P2_ENTREZ_ID %in% ids_disgenet,]
dim(hippie_in_disg)

#(5) Veo como quedaron los scores ----
hist(
  hippie_in_disg[,SCORE],
  xlab = "Score del enlace de HIPPIE",
  ylab = "Count",
  col = "#79CDCD",
  main = "Score de los enlaces de HIPPIE que quedaron en DisGeNET"
)
hist(hippie[!P1_ENTREZ_ID %in% ids_disgenet | !P2_ENTREZ_ID %in% ids_disgenet, SCORE], main = "Score de los enlaces de HIPPIE que quedaron fuera de DisGeNET")
hist(hippie[,SCORE], main = "Score de todos los enlaces de HIPPIE")

